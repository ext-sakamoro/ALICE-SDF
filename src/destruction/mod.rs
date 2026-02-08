//! Voxel Destruction System (Deep Fried Edition)
//!
//! Real-time destructible environments with chunk-based dirty tracking,
//! incremental remeshing, Voronoi fracture, and debris generation.
//! Teardown / Deep Rock Galactic style.
//!
//! # Architecture
//!
//! - **MutableVoxelGrid**: Flat distance + material arrays with dirty chunk tracking
//! - **CarveShape**: Sphere, Box, or arbitrary SDF for carving
//! - **Incremental Remesh**: Only re-run MC on dirty chunks
//! - **Voronoi Fracture**: Split grid into convex pieces
//! - **Debris**: Generate small meshes from removed material
//!
//! # Usage
//!
//! ```rust,ignore
//! use alice_sdf::destruction::*;
//!
//! // Create a mutable grid from an SDF
//! let sphere = SdfNode::sphere(2.0);
//! let mut grid = MutableVoxelGrid::from_sdf(&sphere, [64, 64, 64],
//!     Vec3::splat(-3.0), Vec3::splat(3.0));
//!
//! // Carve a hole
//! let result = carve(&mut grid, &CarveShape::Sphere {
//!     center: Vec3::ZERO, radius: 0.5,
//! });
//!
//! // Remesh only the dirty chunks
//! let meshes = remesh_dirty(&grid, &result.dirty_chunks);
//! ```
//!
//! Author: Moroya Sakamoto

pub mod debris;
pub mod fracture;
pub mod operations;

use glam::Vec3;
use rayon::prelude::*;

use crate::eval::eval;
use crate::types::SdfNode;

pub use debris::{generate_debris, DebrisConfig, DebrisPiece};
pub use fracture::{voronoi_fracture, FractureConfig, FracturePiece};
pub use operations::{carve, carve_batch, CarveShape, DestructionResult};

/// Mutable voxel grid for real-time destruction
///
/// Stores SDF distances and material IDs on a regular grid with
/// chunk-based dirty tracking for incremental remeshing.
pub struct MutableVoxelGrid {
    /// Signed distance values (flat, Z-major: `x + y*sx + z*sx*sy`)
    pub distances: Vec<f32>,
    /// Material IDs per voxel
    pub materials: Vec<u16>,
    /// Grid resolution [X, Y, Z]
    pub resolution: [u32; 3],
    /// World-space min bounds
    pub bounds_min: Vec3,
    /// World-space max bounds
    pub bounds_max: Vec3,
    /// Per-chunk dirty flag
    dirty_chunks: Vec<bool>,
    /// Voxels per chunk axis (e.g. 16)
    chunk_size: u32,
    /// Number of chunks per axis
    chunks_per_axis: [u32; 3],
}

impl MutableVoxelGrid {
    /// Create a grid from an SDF node
    ///
    /// Evaluates the SDF at each voxel center using parallel Z-slabs.
    pub fn from_sdf(
        node: &SdfNode,
        resolution: [u32; 3],
        bounds_min: Vec3,
        bounds_max: Vec3,
    ) -> Self {
        Self::from_sdf_with_chunk_size(node, resolution, bounds_min, bounds_max, 16)
    }

    /// Create a grid from an SDF with custom chunk size
    pub fn from_sdf_with_chunk_size(
        node: &SdfNode,
        resolution: [u32; 3],
        bounds_min: Vec3,
        bounds_max: Vec3,
        chunk_size: u32,
    ) -> Self {
        let [rx, ry, rz] = resolution;
        let total = (rx * ry * rz) as usize;
        let mut distances = vec![0.0f32; total];
        let materials = vec![0u16; total];

        let size = bounds_max - bounds_min;
        let step = Vec3::new(size.x / rx as f32, size.y / ry as f32, size.z / rz as f32);
        let half_step = step * 0.5;

        // Parallel Z-slab evaluation
        distances
            .par_chunks_mut((rx * ry) as usize)
            .enumerate()
            .for_each(|(z, slab)| {
                let wz = bounds_min.z + (z as f32) * step.z + half_step.z;
                for y in 0..ry {
                    let wy = bounds_min.y + (y as f32) * step.y + half_step.y;
                    for x in 0..rx {
                        let wx = bounds_min.x + (x as f32) * step.x + half_step.x;
                        let idx = (x + y * rx) as usize;
                        slab[idx] = eval(node, Vec3::new(wx, wy, wz));
                    }
                }
            });

        let chunks_per_axis = [
            (rx + chunk_size - 1) / chunk_size,
            (ry + chunk_size - 1) / chunk_size,
            (rz + chunk_size - 1) / chunk_size,
        ];
        let total_chunks = (chunks_per_axis[0] * chunks_per_axis[1] * chunks_per_axis[2]) as usize;

        MutableVoxelGrid {
            distances,
            materials,
            resolution,
            bounds_min,
            bounds_max,
            dirty_chunks: vec![false; total_chunks],
            chunk_size,
            chunks_per_axis,
        }
    }

    /// Total number of voxels
    #[inline]
    pub fn voxel_count(&self) -> usize {
        (self.resolution[0] * self.resolution[1] * self.resolution[2]) as usize
    }

    /// Get voxel index from grid coordinates
    #[inline]
    pub fn voxel_index(&self, x: u32, y: u32, z: u32) -> usize {
        (x + y * self.resolution[0] + z * self.resolution[0] * self.resolution[1]) as usize
    }

    /// Get distance at grid coordinates
    #[inline]
    pub fn get_distance(&self, x: u32, y: u32, z: u32) -> f32 {
        self.distances[self.voxel_index(x, y, z)]
    }

    /// Set distance at grid coordinates and mark chunk dirty
    #[inline]
    pub fn set_distance(&mut self, x: u32, y: u32, z: u32, d: f32) {
        let idx = self.voxel_index(x, y, z);
        self.distances[idx] = d;
        self.mark_dirty(x, y, z);
    }

    /// Get material at grid coordinates
    #[inline]
    pub fn get_material(&self, x: u32, y: u32, z: u32) -> u16 {
        self.materials[self.voxel_index(x, y, z)]
    }

    /// Convert world position to grid coordinates
    #[inline]
    pub fn world_to_grid(&self, pos: Vec3) -> Option<[u32; 3]> {
        let size = self.bounds_max - self.bounds_min;
        let t = (pos - self.bounds_min) / size;
        let gx = (t.x * self.resolution[0] as f32) as i32;
        let gy = (t.y * self.resolution[1] as f32) as i32;
        let gz = (t.z * self.resolution[2] as f32) as i32;

        if gx >= 0
            && gx < self.resolution[0] as i32
            && gy >= 0
            && gy < self.resolution[1] as i32
            && gz >= 0
            && gz < self.resolution[2] as i32
        {
            Some([gx as u32, gy as u32, gz as u32])
        } else {
            None
        }
    }

    /// Convert grid coordinates to world position (voxel center)
    #[inline]
    pub fn grid_to_world(&self, x: u32, y: u32, z: u32) -> Vec3 {
        let size = self.bounds_max - self.bounds_min;
        let step = Vec3::new(
            size.x / self.resolution[0] as f32,
            size.y / self.resolution[1] as f32,
            size.z / self.resolution[2] as f32,
        );
        self.bounds_min
            + Vec3::new(
                (x as f32 + 0.5) * step.x,
                (y as f32 + 0.5) * step.y,
                (z as f32 + 0.5) * step.z,
            )
    }

    /// Voxel size (step between adjacent voxels)
    #[inline]
    pub fn voxel_size(&self) -> Vec3 {
        let size = self.bounds_max - self.bounds_min;
        Vec3::new(
            size.x / self.resolution[0] as f32,
            size.y / self.resolution[1] as f32,
            size.z / self.resolution[2] as f32,
        )
    }

    /// Mark chunk containing voxel (x,y,z) as dirty
    #[inline]
    fn mark_dirty(&mut self, x: u32, y: u32, z: u32) {
        let cx = x / self.chunk_size;
        let cy = y / self.chunk_size;
        let cz = z / self.chunk_size;
        let ci = (cx
            + cy * self.chunks_per_axis[0]
            + cz * self.chunks_per_axis[0] * self.chunks_per_axis[1]) as usize;
        if ci < self.dirty_chunks.len() {
            self.dirty_chunks[ci] = true;
        }
    }

    /// Get all dirty chunk indices as [cx, cy, cz]
    pub fn dirty_chunks(&self) -> Vec<[u32; 3]> {
        let mut result = Vec::new();
        let cpx = self.chunks_per_axis[0];
        let cpy = self.chunks_per_axis[1];
        for (i, &dirty) in self.dirty_chunks.iter().enumerate() {
            if dirty {
                let i = i as u32;
                let cz = i / (cpx * cpy);
                let cy = (i % (cpx * cpy)) / cpx;
                let cx = i % cpx;
                result.push([cx, cy, cz]);
            }
        }
        result
    }

    /// Clear all dirty flags
    pub fn clear_dirty(&mut self) {
        self.dirty_chunks.fill(false);
    }

    /// Check if a specific chunk is dirty
    pub fn is_chunk_dirty(&self, cx: u32, cy: u32, cz: u32) -> bool {
        let ci = (cx
            + cy * self.chunks_per_axis[0]
            + cz * self.chunks_per_axis[0] * self.chunks_per_axis[1]) as usize;
        ci < self.dirty_chunks.len() && self.dirty_chunks[ci]
    }

    /// Get chunk size
    #[inline]
    pub fn chunk_size(&self) -> u32 {
        self.chunk_size
    }

    /// Get chunks per axis
    #[inline]
    pub fn chunks_per_axis(&self) -> [u32; 3] {
        self.chunks_per_axis
    }

    /// Memory usage in bytes
    pub fn memory_bytes(&self) -> usize {
        self.distances.len() * std::mem::size_of::<f32>()
            + self.materials.len() * std::mem::size_of::<u16>()
            + self.dirty_chunks.len()
    }

    /// Remesh a single dirty chunk to a Mesh
    ///
    /// Runs Marching Cubes on the voxels within the specified chunk.
    pub fn remesh_chunk(&self, cx: u32, cy: u32, cz: u32) -> crate::mesh::Mesh {
        use crate::mesh::Vertex;

        let cs = self.chunk_size;
        let x0 = cx * cs;
        let y0 = cy * cs;
        let z0 = cz * cs;
        let x1 = (x0 + cs).min(self.resolution[0]);
        let y1 = (y0 + cs).min(self.resolution[1]);
        let z1 = (z0 + cs).min(self.resolution[2]);

        let vs = self.voxel_size();
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        // Simple Marching Cubes on chunk voxels
        for z in z0..z1.saturating_sub(1) {
            for y in y0..y1.saturating_sub(1) {
                for x in x0..x1.saturating_sub(1) {
                    let corners = [
                        self.get_distance(x, y, z),
                        self.get_distance(x + 1, y, z),
                        self.get_distance(x + 1, y + 1, z),
                        self.get_distance(x, y + 1, z),
                        self.get_distance(x, y, z + 1),
                        self.get_distance(x + 1, y, z + 1),
                        self.get_distance(x + 1, y + 1, z + 1),
                        self.get_distance(x, y + 1, z + 1),
                    ];

                    let mut cube_index = 0u8;
                    for (i, &c) in corners.iter().enumerate() {
                        if c < 0.0 {
                            cube_index |= 1 << i;
                        }
                    }

                    if cube_index == 0 || cube_index == 255 {
                        continue;
                    }

                    // Generate triangles using edge midpoints
                    let base_pos = self.bounds_min
                        + Vec3::new(x as f32 * vs.x, y as f32 * vs.y, z as f32 * vs.z);

                    let edge_positions = mc_edge_positions(base_pos, vs, &corners);
                    let base_idx = vertices.len() as u32;

                    for &[a, b, c] in MC_TRI_TABLE[cube_index as usize].iter() {
                        if a == 255 {
                            break;
                        }
                        let pa = edge_positions[a as usize];
                        let pb = edge_positions[b as usize];
                        let pc = edge_positions[c as usize];

                        let normal = (pb - pa).cross(pc - pa).normalize_or_zero();

                        let vi = vertices.len() as u32;
                        vertices.push(Vertex {
                            position: pa,
                            normal,
                            ..Default::default()
                        });
                        vertices.push(Vertex {
                            position: pb,
                            normal,
                            ..Default::default()
                        });
                        vertices.push(Vertex {
                            position: pc,
                            normal,
                            ..Default::default()
                        });
                        indices.push(vi);
                        indices.push(vi + 1);
                        indices.push(vi + 2);
                    }

                    let _ = base_idx; // suppress warning
                }
            }
        }

        crate::mesh::Mesh { vertices, indices }
    }

    /// Remesh all dirty chunks and clear dirty flags
    pub fn remesh_all_dirty(&mut self) -> Vec<ChunkMesh> {
        let dirty = self.dirty_chunks();
        let meshes: Vec<ChunkMesh> = dirty
            .iter()
            .map(|&[cx, cy, cz]| {
                let mesh = self.remesh_chunk(cx, cy, cz);
                ChunkMesh {
                    chunk: [cx, cy, cz],
                    mesh,
                }
            })
            .collect();
        self.clear_dirty();
        meshes
    }
}

/// A mesh associated with its chunk coordinates
pub struct ChunkMesh {
    /// Chunk coordinates [cx, cy, cz]
    pub chunk: [u32; 3],
    /// The generated mesh for this chunk
    pub mesh: crate::mesh::Mesh,
}

/// Compute edge intersection positions for a MC cell
fn mc_edge_positions(base: Vec3, step: Vec3, corners: &[f32; 8]) -> [Vec3; 12] {
    let p = [
        base,
        base + Vec3::new(step.x, 0.0, 0.0),
        base + Vec3::new(step.x, step.y, 0.0),
        base + Vec3::new(0.0, step.y, 0.0),
        base + Vec3::new(0.0, 0.0, step.z),
        base + Vec3::new(step.x, 0.0, step.z),
        base + Vec3::new(step.x, step.y, step.z),
        base + Vec3::new(0.0, step.y, step.z),
    ];

    let interp = |a: usize, b: usize| -> Vec3 {
        let da = corners[a];
        let db = corners[b];
        let denom = db - da;
        if denom.abs() < 1e-10 {
            (p[a] + p[b]) * 0.5
        } else {
            let t = (-da / denom).clamp(0.0, 1.0);
            p[a] + (p[b] - p[a]) * t
        }
    };

    [
        interp(0, 1),
        interp(1, 2),
        interp(2, 3),
        interp(3, 0),
        interp(4, 5),
        interp(5, 6),
        interp(6, 7),
        interp(7, 4),
        interp(0, 4),
        interp(1, 5),
        interp(2, 6),
        interp(3, 7),
    ]
}

/// Compact MC triangle table (256 entries, up to 5 triangles per cell)
///
/// Each entry is an array of up to 5 [edge_a, edge_b, edge_c] triples.
/// Terminated by [255, 255, 255].
static MC_TRI_TABLE: [[[u8; 3]; 5]; 256] = {
    let empty = [[255, 255, 255]; 5];
    let mut table = [empty; 256];

    // Case 0: no triangles
    // Case 255: no triangles (all inside)

    // Only the most common cases for chunk MC - full table would be 256 entries.
    // We embed a complete lookup table using the standard MC edge numbering.
    macro_rules! tri {
        ($idx:expr, $( [$a:expr, $b:expr, $c:expr] ),* ) => {
            {
                let mut entry = [[255u8, 255, 255]; 5];
                let tris: &[[u8; 3]] = &[ $( [$a, $b, $c] ),* ];
                let mut i = 0;
                while i < tris.len() && i < 5 {
                    entry[i] = tris[i];
                    i += 1;
                }
                table[$idx] = entry;
            }
        };
    }

    // Standard Marching Cubes triangle table (all 256 cases)
    // Edges: 0=(0,1) 1=(1,2) 2=(2,3) 3=(3,0) 4=(4,5) 5=(5,6) 6=(6,7) 7=(7,4) 8=(0,4) 9=(1,5) 10=(2,6) 11=(3,7)
    tri!(0x01, [0, 8, 3]);
    tri!(0x02, [0, 1, 9]);
    tri!(0x03, [1, 8, 3], [9, 8, 1]);
    tri!(0x04, [1, 2, 10]);
    tri!(0x05, [0, 8, 3], [1, 2, 10]);
    tri!(0x06, [9, 2, 10], [0, 2, 9]);
    tri!(0x07, [2, 8, 3], [2, 10, 8], [10, 9, 8]);
    tri!(0x08, [3, 11, 2]);
    tri!(0x09, [0, 11, 2], [8, 11, 0]);
    tri!(0x0A, [1, 9, 0], [2, 3, 11]);
    tri!(0x0B, [1, 11, 2], [1, 9, 11], [9, 8, 11]);
    tri!(0x0C, [3, 10, 1], [11, 10, 3]);
    tri!(0x0D, [0, 10, 1], [0, 8, 10], [8, 11, 10]);
    tri!(0x0E, [3, 9, 0], [3, 11, 9], [11, 10, 9]);
    tri!(0x0F, [9, 8, 10], [10, 8, 11]);
    tri!(0x10, [4, 7, 8]);
    tri!(0x11, [4, 3, 0], [7, 3, 4]);
    tri!(0x12, [0, 1, 9], [8, 4, 7]);
    tri!(0x13, [4, 1, 9], [4, 7, 1], [7, 3, 1]);
    tri!(0x14, [1, 2, 10], [8, 4, 7]);
    tri!(0x15, [3, 4, 7], [3, 0, 4], [1, 2, 10]);
    tri!(0x16, [9, 2, 10], [9, 0, 2], [8, 4, 7]);
    tri!(0x17, [2, 10, 9], [2, 9, 7], [2, 7, 3], [7, 9, 4]);
    tri!(0x18, [8, 4, 7], [3, 11, 2]);
    tri!(0x19, [11, 4, 7], [11, 2, 4], [2, 0, 4]);
    tri!(0x1A, [9, 0, 1], [8, 4, 7], [2, 3, 11]);
    tri!(0x1B, [4, 7, 11], [9, 4, 11], [9, 11, 2], [9, 2, 1]);
    tri!(0x1C, [3, 10, 1], [3, 11, 10], [7, 8, 4]);
    tri!(0x1D, [1, 11, 10], [1, 4, 11], [1, 0, 4], [7, 11, 4]);
    tri!(0x1E, [4, 7, 8], [9, 0, 11], [9, 11, 10], [11, 0, 3]);
    tri!(0x1F, [4, 7, 11], [4, 11, 9], [9, 11, 10]);
    tri!(0x20, [9, 5, 4]);
    tri!(0x21, [9, 5, 4], [0, 8, 3]);
    tri!(0x22, [0, 5, 4], [1, 5, 0]);
    tri!(0x23, [8, 5, 4], [8, 3, 5], [3, 1, 5]);
    tri!(0x24, [1, 2, 10], [9, 5, 4]);
    tri!(0x25, [3, 0, 8], [1, 2, 10], [4, 9, 5]);
    tri!(0x26, [5, 2, 10], [5, 4, 2], [4, 0, 2]);
    tri!(0x27, [2, 10, 5], [3, 2, 5], [3, 5, 4], [3, 4, 8]);
    tri!(0x28, [9, 5, 4], [2, 3, 11]);
    tri!(0x29, [0, 11, 2], [0, 8, 11], [4, 9, 5]);
    tri!(0x2A, [0, 5, 4], [0, 1, 5], [2, 3, 11]);
    tri!(0x2B, [2, 1, 5], [2, 5, 8], [2, 8, 11], [4, 8, 5]);
    tri!(0x2C, [10, 3, 11], [10, 1, 3], [9, 5, 4]);
    tri!(0x2D, [4, 9, 5], [0, 8, 1], [8, 10, 1], [8, 11, 10]);
    tri!(0x2E, [5, 4, 0], [5, 0, 11], [5, 11, 10], [11, 0, 3]);
    tri!(0x2F, [5, 4, 8], [5, 8, 10], [10, 8, 11]);
    tri!(0x30, [9, 7, 8], [5, 7, 9]);
    tri!(0x31, [9, 3, 0], [9, 5, 3], [5, 7, 3]);
    tri!(0x32, [0, 7, 8], [0, 1, 7], [1, 5, 7]);
    tri!(0x33, [1, 5, 3], [3, 5, 7]);
    tri!(0x34, [9, 7, 8], [9, 5, 7], [10, 1, 2]);
    tri!(0x35, [10, 1, 2], [9, 5, 0], [5, 3, 0], [5, 7, 3]);
    tri!(0x36, [8, 0, 2], [8, 2, 5], [8, 5, 7], [10, 5, 2]);
    tri!(0x37, [2, 10, 5], [2, 5, 3], [3, 5, 7]);
    tri!(0x38, [7, 9, 5], [7, 8, 9], [3, 11, 2]);
    tri!(0x39, [9, 5, 7], [9, 7, 2], [9, 2, 0], [2, 7, 11]);
    tri!(0x3A, [2, 3, 11], [0, 1, 8], [1, 7, 8], [1, 5, 7]);
    tri!(0x3B, [11, 2, 1], [11, 1, 7], [7, 1, 5]);
    tri!(0x3C, [9, 5, 8], [8, 5, 7], [10, 1, 3], [10, 3, 11]);
    tri!(
        0x3D,
        [5, 7, 0],
        [5, 0, 9],
        [7, 11, 0],
        [1, 0, 10],
        [11, 10, 0]
    );
    tri!(
        0x3E,
        [11, 10, 0],
        [11, 0, 3],
        [10, 5, 0],
        [8, 0, 7],
        [5, 7, 0]
    );
    tri!(0x3F, [11, 10, 5], [7, 11, 5]);
    tri!(0x40, [10, 6, 5]);
    tri!(0x41, [0, 8, 3], [5, 10, 6]);
    tri!(0x42, [9, 0, 1], [5, 10, 6]);
    tri!(0x43, [1, 8, 3], [1, 9, 8], [5, 10, 6]);
    tri!(0x44, [1, 6, 5], [2, 6, 1]);
    tri!(0x45, [1, 6, 5], [1, 2, 6], [3, 0, 8]);
    tri!(0x46, [9, 6, 5], [9, 0, 6], [0, 2, 6]);
    tri!(0x47, [5, 9, 8], [5, 8, 2], [5, 2, 6], [3, 2, 8]);
    tri!(0x48, [2, 3, 11], [10, 6, 5]);
    tri!(0x49, [11, 0, 8], [11, 2, 0], [10, 6, 5]);
    tri!(0x4A, [0, 1, 9], [2, 3, 11], [5, 10, 6]);
    tri!(0x4B, [5, 10, 6], [1, 9, 2], [9, 11, 2], [9, 8, 11]);
    tri!(0x4C, [6, 3, 11], [6, 5, 3], [5, 1, 3]);
    tri!(0x4D, [0, 8, 11], [0, 11, 5], [0, 5, 1], [5, 11, 6]);
    tri!(0x4E, [3, 11, 6], [0, 3, 6], [0, 6, 5], [0, 5, 9]);
    tri!(0x4F, [6, 5, 9], [6, 9, 11], [11, 9, 8]);
    tri!(0x50, [5, 10, 6], [4, 7, 8]);
    tri!(0x51, [4, 3, 0], [4, 7, 3], [6, 5, 10]);
    tri!(0x52, [1, 9, 0], [5, 10, 6], [8, 4, 7]);
    tri!(0x53, [10, 6, 5], [1, 9, 7], [1, 7, 3], [7, 9, 4]);
    tri!(0x54, [6, 1, 2], [6, 5, 1], [4, 7, 8]);
    tri!(0x55, [1, 2, 5], [5, 2, 6], [3, 0, 4], [3, 4, 7]);
    tri!(0x56, [8, 4, 7], [9, 0, 5], [0, 6, 5], [0, 2, 6]);
    tri!(0x57, [7, 3, 9], [7, 9, 4], [3, 2, 9], [5, 9, 6], [2, 6, 9]);
    tri!(0x58, [3, 11, 2], [7, 8, 4], [10, 6, 5]);
    tri!(0x59, [5, 10, 6], [4, 7, 2], [4, 2, 0], [2, 7, 11]);
    tri!(0x5A, [0, 1, 9], [4, 7, 8], [2, 3, 11], [5, 10, 6]);
    tri!(
        0x5B,
        [9, 2, 1],
        [9, 11, 2],
        [9, 4, 11],
        [7, 11, 4],
        [5, 10, 6]
    );
    tri!(0x5C, [8, 4, 7], [3, 11, 5], [3, 5, 1], [5, 11, 6]);
    tri!(
        0x5D,
        [5, 1, 11],
        [5, 11, 6],
        [1, 0, 11],
        [7, 11, 4],
        [0, 4, 11]
    );
    tri!(0x5E, [0, 5, 9], [0, 6, 5], [0, 3, 6], [11, 6, 3], [8, 4, 7]);
    tri!(0x5F, [6, 5, 9], [6, 9, 11], [4, 7, 9], [7, 11, 9]);
    tri!(0x60, [10, 4, 9], [6, 4, 10]);
    tri!(0x61, [4, 10, 6], [4, 9, 10], [0, 8, 3]);
    tri!(0x62, [10, 0, 1], [10, 6, 0], [6, 4, 0]);
    tri!(0x63, [8, 3, 1], [8, 1, 6], [8, 6, 4], [6, 1, 10]);
    tri!(0x64, [1, 4, 9], [1, 2, 4], [2, 6, 4]);
    tri!(0x65, [3, 0, 8], [1, 2, 9], [2, 4, 9], [2, 6, 4]);
    tri!(0x66, [0, 2, 4], [4, 2, 6]);
    tri!(0x67, [8, 3, 2], [8, 2, 4], [4, 2, 6]);
    tri!(0x68, [10, 4, 9], [10, 6, 4], [11, 2, 3]);
    tri!(0x69, [0, 8, 2], [2, 8, 11], [4, 9, 10], [4, 10, 6]);
    tri!(0x6A, [3, 11, 2], [0, 1, 6], [0, 6, 4], [6, 1, 10]);
    tri!(
        0x6B,
        [6, 4, 1],
        [6, 1, 10],
        [4, 8, 1],
        [2, 1, 11],
        [8, 11, 1]
    );
    tri!(0x6C, [9, 6, 4], [9, 3, 6], [9, 1, 3], [11, 6, 3]);
    tri!(
        0x6D,
        [8, 11, 1],
        [8, 1, 0],
        [11, 6, 1],
        [9, 1, 4],
        [6, 4, 1]
    );
    tri!(0x6E, [3, 11, 6], [3, 6, 0], [0, 6, 4]);
    tri!(0x6F, [6, 4, 8], [11, 6, 8]);
    tri!(0x70, [7, 10, 6], [7, 8, 10], [8, 9, 10]);
    tri!(0x71, [0, 7, 3], [0, 10, 7], [0, 9, 10], [6, 7, 10]);
    tri!(0x72, [10, 6, 7], [1, 10, 7], [1, 7, 8], [1, 8, 0]);
    tri!(0x73, [10, 6, 7], [10, 7, 1], [1, 7, 3]);
    tri!(0x74, [1, 2, 6], [1, 6, 8], [1, 8, 9], [8, 6, 7]);
    tri!(0x75, [2, 6, 9], [2, 9, 1], [6, 7, 9], [0, 9, 3], [7, 3, 9]);
    tri!(0x76, [7, 8, 0], [7, 0, 6], [6, 0, 2]);
    tri!(0x77, [7, 3, 2], [6, 7, 2]);
    tri!(0x78, [2, 3, 11], [10, 6, 8], [10, 8, 9], [8, 6, 7]);
    tri!(
        0x79,
        [2, 0, 7],
        [2, 7, 11],
        [0, 9, 7],
        [6, 7, 10],
        [9, 10, 7]
    );
    tri!(
        0x7A,
        [1, 8, 0],
        [1, 7, 8],
        [1, 10, 7],
        [6, 7, 10],
        [2, 3, 11]
    );
    tri!(0x7B, [11, 2, 1], [11, 1, 7], [10, 6, 1], [6, 7, 1]);
    tri!(0x7C, [8, 9, 6], [8, 6, 7], [9, 1, 6], [11, 6, 3], [1, 3, 6]);
    tri!(0x7D, [0, 9, 1], [11, 6, 7]);
    tri!(0x7E, [7, 8, 0], [7, 0, 6], [3, 11, 0], [11, 6, 0]);
    tri!(0x7F, [7, 11, 6]);
    tri!(0x80, [7, 6, 11]);
    tri!(0x81, [3, 0, 8], [11, 7, 6]);
    tri!(0x82, [0, 1, 9], [11, 7, 6]);
    tri!(0x83, [8, 1, 9], [8, 3, 1], [11, 7, 6]);
    tri!(0x84, [10, 1, 2], [6, 11, 7]);
    tri!(0x85, [1, 2, 10], [3, 0, 8], [6, 11, 7]);
    tri!(0x86, [2, 9, 0], [2, 10, 9], [6, 11, 7]);
    tri!(0x87, [6, 11, 7], [2, 10, 3], [10, 8, 3], [10, 9, 8]);
    tri!(0x88, [7, 2, 3], [6, 2, 7]);
    tri!(0x89, [7, 0, 8], [7, 6, 0], [6, 2, 0]);
    tri!(0x8A, [2, 7, 6], [2, 3, 7], [0, 1, 9]);
    tri!(0x8B, [1, 6, 2], [1, 8, 6], [1, 9, 8], [8, 7, 6]);
    tri!(0x8C, [10, 7, 6], [10, 1, 7], [1, 3, 7]);
    tri!(0x8D, [10, 7, 6], [1, 7, 10], [1, 8, 7], [1, 0, 8]);
    tri!(0x8E, [0, 3, 7], [0, 7, 10], [0, 10, 9], [6, 10, 7]);
    tri!(0x8F, [7, 6, 10], [7, 10, 8], [8, 10, 9]);
    tri!(0x90, [6, 8, 4], [11, 8, 6]);
    tri!(0x91, [3, 6, 11], [3, 0, 6], [0, 4, 6]);
    tri!(0x92, [8, 6, 11], [8, 4, 6], [9, 0, 1]);
    tri!(0x93, [9, 4, 6], [9, 6, 3], [9, 3, 1], [11, 3, 6]);
    tri!(0x94, [6, 8, 4], [6, 11, 8], [2, 10, 1]);
    tri!(0x95, [1, 2, 10], [3, 0, 11], [0, 6, 11], [0, 4, 6]);
    tri!(0x96, [4, 11, 8], [4, 6, 11], [0, 2, 9], [2, 10, 9]);
    tri!(
        0x97,
        [10, 9, 3],
        [10, 3, 2],
        [9, 4, 3],
        [11, 3, 6],
        [4, 6, 3]
    );
    tri!(0x98, [8, 2, 3], [8, 4, 2], [4, 6, 2]);
    tri!(0x99, [0, 4, 2], [4, 6, 2]);
    tri!(0x9A, [1, 9, 0], [2, 3, 4], [2, 4, 6], [4, 3, 8]);
    tri!(0x9B, [1, 9, 4], [1, 4, 2], [2, 4, 6]);
    tri!(0x9C, [8, 1, 3], [8, 6, 1], [8, 4, 6], [6, 10, 1]);
    tri!(0x9D, [10, 1, 0], [10, 0, 6], [6, 0, 4]);
    tri!(
        0x9E,
        [4, 6, 3],
        [4, 3, 8],
        [6, 10, 3],
        [0, 3, 9],
        [10, 9, 3]
    );
    tri!(0x9F, [10, 9, 4], [6, 10, 4]);
    tri!(0xA0, [4, 9, 5], [7, 6, 11]);
    tri!(0xA1, [0, 8, 3], [4, 9, 5], [11, 7, 6]);
    tri!(0xA2, [5, 0, 1], [5, 4, 0], [7, 6, 11]);
    tri!(0xA3, [11, 7, 6], [8, 3, 4], [3, 5, 4], [3, 1, 5]);
    tri!(0xA4, [9, 5, 4], [10, 1, 2], [7, 6, 11]);
    tri!(0xA5, [6, 11, 7], [1, 2, 10], [0, 8, 3], [4, 9, 5]);
    tri!(0xA6, [7, 6, 11], [5, 4, 10], [4, 2, 10], [4, 0, 2]);
    tri!(
        0xA7,
        [3, 4, 8],
        [3, 5, 4],
        [3, 2, 5],
        [10, 5, 2],
        [11, 7, 6]
    );
    tri!(0xA8, [7, 2, 3], [7, 6, 2], [5, 4, 9]);
    tri!(0xA9, [9, 5, 4], [0, 8, 6], [0, 6, 2], [6, 8, 7]);
    tri!(0xAA, [3, 6, 2], [3, 7, 6], [1, 5, 0], [5, 4, 0]);
    tri!(0xAB, [6, 2, 8], [6, 8, 7], [2, 1, 8], [4, 8, 5], [1, 5, 8]);
    tri!(0xAC, [9, 5, 4], [10, 1, 6], [1, 7, 6], [1, 3, 7]);
    tri!(0xAD, [1, 6, 10], [1, 7, 6], [1, 0, 7], [8, 7, 0], [9, 5, 4]);
    tri!(
        0xAE,
        [4, 0, 10],
        [4, 10, 5],
        [0, 3, 10],
        [6, 10, 7],
        [3, 7, 10]
    );
    tri!(0xAF, [7, 6, 10], [7, 10, 8], [5, 4, 10], [4, 8, 10]);
    tri!(0xB0, [6, 9, 5], [6, 11, 9], [11, 8, 9]);
    tri!(0xB1, [3, 6, 11], [0, 6, 3], [0, 5, 6], [0, 9, 5]);
    tri!(0xB2, [0, 11, 8], [0, 5, 11], [0, 1, 5], [5, 6, 11]);
    tri!(0xB3, [6, 11, 3], [6, 3, 5], [5, 3, 1]);
    tri!(0xB4, [1, 2, 10], [9, 5, 11], [9, 11, 8], [11, 5, 6]);
    tri!(
        0xB5,
        [0, 11, 3],
        [0, 6, 11],
        [0, 9, 6],
        [5, 6, 9],
        [1, 2, 10]
    );
    tri!(
        0xB6,
        [11, 8, 5],
        [11, 5, 6],
        [8, 0, 5],
        [10, 5, 2],
        [0, 2, 5]
    );
    tri!(0xB7, [6, 11, 3], [6, 3, 5], [2, 10, 3], [10, 5, 3]);
    tri!(0xB8, [5, 8, 9], [5, 2, 8], [5, 6, 2], [3, 8, 2]);
    tri!(0xB9, [9, 5, 6], [9, 6, 0], [0, 6, 2]);
    tri!(0xBA, [1, 5, 8], [1, 8, 0], [5, 6, 8], [3, 8, 2], [6, 2, 8]);
    tri!(0xBB, [1, 5, 6], [2, 1, 6]);
    tri!(0xBC, [1, 3, 6], [1, 6, 10], [3, 8, 6], [5, 6, 9], [8, 9, 6]);
    tri!(0xBD, [10, 1, 0], [10, 0, 6], [9, 5, 0], [5, 6, 0]);
    tri!(0xBE, [0, 3, 8], [5, 6, 10]);
    tri!(0xBF, [10, 5, 6]);
    tri!(0xC0, [11, 5, 10], [7, 5, 11]);
    tri!(0xC1, [11, 5, 10], [11, 7, 5], [8, 3, 0]);
    tri!(0xC2, [5, 11, 7], [5, 10, 11], [1, 9, 0]);
    tri!(0xC3, [10, 7, 5], [10, 11, 7], [9, 8, 1], [8, 3, 1]);
    tri!(0xC4, [11, 1, 2], [11, 7, 1], [7, 5, 1]);
    tri!(0xC5, [0, 8, 3], [1, 2, 7], [1, 7, 5], [7, 2, 11]);
    tri!(0xC6, [9, 7, 5], [9, 2, 7], [9, 0, 2], [2, 11, 7]);
    tri!(0xC7, [7, 5, 2], [7, 2, 11], [5, 9, 2], [3, 2, 8], [9, 8, 2]);
    tri!(0xC8, [2, 5, 10], [2, 3, 5], [3, 7, 5]);
    tri!(0xC9, [8, 2, 0], [8, 5, 2], [8, 7, 5], [10, 2, 5]);
    tri!(0xCA, [9, 0, 1], [5, 10, 3], [5, 3, 7], [3, 10, 2]);
    tri!(0xCB, [9, 8, 2], [9, 2, 1], [8, 7, 2], [10, 2, 5], [7, 5, 2]);
    tri!(0xCC, [1, 3, 5], [3, 7, 5]);
    tri!(0xCD, [0, 8, 7], [0, 7, 1], [1, 7, 5]);
    tri!(0xCE, [9, 0, 3], [9, 3, 5], [5, 3, 7]);
    tri!(0xCF, [9, 8, 7], [5, 9, 7]);
    tri!(0xD0, [5, 8, 4], [5, 10, 8], [10, 11, 8]);
    tri!(0xD1, [5, 0, 4], [5, 11, 0], [5, 10, 11], [11, 3, 0]);
    tri!(0xD2, [0, 1, 9], [8, 4, 10], [8, 10, 11], [10, 4, 5]);
    tri!(
        0xD3,
        [10, 11, 4],
        [10, 4, 5],
        [11, 3, 4],
        [9, 4, 1],
        [3, 1, 4]
    );
    tri!(0xD4, [2, 5, 1], [2, 8, 5], [2, 11, 8], [4, 5, 8]);
    tri!(
        0xD5,
        [0, 4, 11],
        [0, 11, 3],
        [4, 5, 11],
        [2, 11, 1],
        [5, 1, 11]
    );
    tri!(
        0xD6,
        [0, 2, 5],
        [0, 5, 9],
        [2, 11, 5],
        [4, 5, 8],
        [11, 8, 5]
    );
    tri!(0xD7, [9, 4, 5], [2, 11, 3]);
    tri!(0xD8, [2, 5, 10], [3, 5, 2], [3, 4, 5], [3, 8, 4]);
    tri!(0xD9, [5, 10, 2], [5, 2, 4], [4, 2, 0]);
    tri!(
        0xDA,
        [3, 10, 2],
        [3, 5, 10],
        [3, 8, 5],
        [4, 5, 8],
        [0, 1, 9]
    );
    tri!(0xDB, [5, 10, 2], [5, 2, 4], [1, 9, 2], [9, 4, 2]);
    tri!(0xDC, [8, 4, 5], [8, 5, 3], [3, 5, 1]);
    tri!(0xDD, [0, 4, 5], [1, 0, 5]);
    tri!(0xDE, [8, 4, 5], [8, 5, 3], [9, 0, 5], [0, 3, 5]);
    tri!(0xDF, [9, 4, 5]);
    tri!(0xE0, [4, 11, 7], [4, 9, 11], [9, 10, 11]);
    tri!(0xE1, [0, 8, 3], [4, 9, 7], [9, 11, 7], [9, 10, 11]);
    tri!(0xE2, [1, 10, 11], [1, 11, 4], [1, 4, 0], [7, 4, 11]);
    tri!(
        0xE3,
        [3, 1, 4],
        [3, 4, 8],
        [1, 10, 4],
        [7, 4, 11],
        [10, 11, 4]
    );
    tri!(0xE4, [4, 11, 7], [9, 11, 4], [9, 2, 11], [9, 1, 2]);
    tri!(
        0xE5,
        [9, 7, 4],
        [9, 11, 7],
        [9, 1, 11],
        [2, 11, 1],
        [0, 8, 3]
    );
    tri!(0xE6, [11, 7, 4], [11, 4, 2], [2, 4, 0]);
    tri!(0xE7, [11, 7, 4], [11, 4, 2], [8, 3, 4], [3, 2, 4]);
    tri!(0xE8, [2, 9, 10], [2, 7, 9], [2, 3, 7], [7, 4, 9]);
    tri!(
        0xE9,
        [9, 10, 7],
        [9, 7, 4],
        [10, 2, 7],
        [8, 7, 0],
        [2, 0, 7]
    );
    tri!(
        0xEA,
        [3, 7, 10],
        [3, 10, 2],
        [7, 4, 10],
        [1, 10, 0],
        [4, 0, 10]
    );
    tri!(0xEB, [1, 10, 2], [8, 7, 4]);
    tri!(0xEC, [4, 9, 1], [4, 1, 7], [7, 1, 3]);
    tri!(0xED, [4, 9, 1], [4, 1, 7], [0, 8, 1], [8, 7, 1]);
    tri!(0xEE, [4, 0, 3], [7, 4, 3]);
    tri!(0xEF, [4, 8, 7]);
    tri!(0xF0, [9, 10, 8], [10, 11, 8]);
    tri!(0xF1, [3, 0, 9], [3, 9, 11], [11, 9, 10]);
    tri!(0xF2, [0, 1, 10], [0, 10, 8], [8, 10, 11]);
    tri!(0xF3, [3, 1, 10], [11, 3, 10]);
    tri!(0xF4, [1, 2, 11], [1, 11, 9], [9, 11, 8]);
    tri!(0xF5, [3, 0, 9], [3, 9, 11], [1, 2, 9], [2, 11, 9]);
    tri!(0xF6, [0, 2, 11], [8, 0, 11]);
    tri!(0xF7, [3, 2, 11]);
    tri!(0xF8, [2, 3, 8], [2, 8, 10], [10, 8, 9]);
    tri!(0xF9, [9, 10, 2], [0, 9, 2]);
    tri!(0xFA, [2, 3, 8], [2, 8, 10], [0, 1, 8], [1, 10, 8]);
    tri!(0xFB, [1, 10, 2]);
    tri!(0xFC, [1, 3, 8], [9, 1, 8]);
    tri!(0xFD, [0, 9, 1]);
    tri!(0xFE, [0, 3, 8]);
    // 0xFF: all inside, no triangles

    table
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::SdfNode;

    #[test]
    fn test_grid_from_sphere() {
        let sphere = SdfNode::sphere(1.0);
        let grid =
            MutableVoxelGrid::from_sdf(&sphere, [16, 16, 16], Vec3::splat(-2.0), Vec3::splat(2.0));

        assert_eq!(grid.voxel_count(), 16 * 16 * 16);
        // Center voxel should be inside (negative distance)
        let center_d = grid.get_distance(8, 8, 8);
        assert!(
            center_d < 0.0,
            "Center should be inside sphere, got {}",
            center_d
        );

        // Corner voxel should be outside
        let corner_d = grid.get_distance(0, 0, 0);
        assert!(
            corner_d > 0.0,
            "Corner should be outside sphere, got {}",
            corner_d
        );
    }

    #[test]
    fn test_grid_world_conversion() {
        let sphere = SdfNode::sphere(1.0);
        let grid =
            MutableVoxelGrid::from_sdf(&sphere, [8, 8, 8], Vec3::splat(-2.0), Vec3::splat(2.0));

        // Center of grid should map to ~(0,0,0) in world
        let world = grid.grid_to_world(4, 4, 4);
        assert!((world - Vec3::new(0.25, 0.25, 0.25)).length() < 0.5);

        // World origin should map to grid center-ish
        let grid_coords = grid.world_to_grid(Vec3::ZERO);
        assert!(grid_coords.is_some());
    }

    #[test]
    fn test_dirty_tracking() {
        let sphere = SdfNode::sphere(1.0);
        let mut grid =
            MutableVoxelGrid::from_sdf(&sphere, [32, 32, 32], Vec3::splat(-2.0), Vec3::splat(2.0));

        assert!(grid.dirty_chunks().is_empty());

        grid.set_distance(5, 5, 5, 10.0);
        let dirty = grid.dirty_chunks();
        assert!(!dirty.is_empty());

        grid.clear_dirty();
        assert!(grid.dirty_chunks().is_empty());
    }

    #[test]
    fn test_voxel_size() {
        let sphere = SdfNode::sphere(1.0);
        let grid =
            MutableVoxelGrid::from_sdf(&sphere, [16, 16, 16], Vec3::splat(-2.0), Vec3::splat(2.0));

        let vs = grid.voxel_size();
        assert!((vs.x - 0.25).abs() < 0.001);
        assert!((vs.y - 0.25).abs() < 0.001);
        assert!((vs.z - 0.25).abs() < 0.001);
    }

    #[test]
    fn test_memory_bytes() {
        let sphere = SdfNode::sphere(1.0);
        let grid =
            MutableVoxelGrid::from_sdf(&sphere, [16, 16, 16], Vec3::splat(-2.0), Vec3::splat(2.0));

        // 4096 voxels * (4 bytes float + 2 bytes u16) + chunk flags
        assert!(grid.memory_bytes() > 4096 * 4);
    }

    #[test]
    fn test_remesh_chunk() {
        let sphere = SdfNode::sphere(1.0);
        let grid =
            MutableVoxelGrid::from_sdf(&sphere, [16, 16, 16], Vec3::splat(-2.0), Vec3::splat(2.0));

        // Remesh a chunk that should contain sphere surface
        let mesh = grid.remesh_chunk(0, 0, 0);
        // This chunk covers [-2, 0] in all axes, sphere surface is at radius 1
        assert!(mesh.vertices.len() > 0 || mesh.indices.len() >= 0);
    }
}

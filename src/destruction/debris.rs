//! Debris Generation: Create mesh fragments from destroyed material (Deep Fried Edition)
//!
//! When material is removed from the voxel grid, this module generates small
//! mesh fragments (debris) that can be used as physics objects.
//!
//! Author: Moroya Sakamoto

use glam::Vec3;

use crate::mesh::{Mesh, Vertex};

/// Configuration for debris generation
#[derive(Debug, Clone)]
pub struct DebrisConfig {
    /// Maximum number of debris pieces
    pub max_pieces: u32,
    /// Minimum debris size (world units)
    pub min_size: f32,
    /// Maximum debris size (world units)
    pub max_size: f32,
    /// Random seed
    pub seed: u64,
}

impl Default for DebrisConfig {
    fn default() -> Self {
        DebrisConfig {
            max_pieces: 8,
            min_size: 0.05,
            max_size: 0.2,
            seed: 42,
        }
    }
}

/// A piece of debris with position and mesh
#[derive(Debug)]
pub struct DebrisPiece {
    /// Center position of the debris in world space
    pub center: Vec3,
    /// Approximate radius of the debris
    pub radius: f32,
    /// The debris mesh
    pub mesh: Mesh,
    /// Approximate volume
    pub volume: f32,
}

/// Generate debris meshes from a carve operation
///
/// Given the carve center, radius, and the old/new distances at affected voxels,
/// generates small convex debris pieces.
pub fn generate_debris(center: Vec3, radius: f32, config: &DebrisConfig) -> Vec<DebrisPiece> {
    let mut pieces = Vec::new();
    let mut rng_state = config.seed;

    let piece_count = config
        .max_pieces
        .min(((radius / config.min_size) as u32).max(1));

    for _ in 0..piece_count {
        // Generate random position within the carve area
        rng_state = lcg_next(rng_state);
        let rx = lcg_float(rng_state) * 2.0 - 1.0;
        rng_state = lcg_next(rng_state);
        let ry = lcg_float(rng_state) * 2.0 - 1.0;
        rng_state = lcg_next(rng_state);
        let rz = lcg_float(rng_state) * 2.0 - 1.0;

        let dir = Vec3::new(rx, ry, rz).normalize_or_zero();
        rng_state = lcg_next(rng_state);
        let dist = lcg_float(rng_state) * radius;
        let piece_center = center + dir * dist;

        rng_state = lcg_next(rng_state);
        let piece_radius =
            config.min_size + lcg_float(rng_state) * (config.max_size - config.min_size);

        // Generate a simple convex debris mesh (distorted icosahedron)
        let mesh = generate_debris_mesh(piece_center, piece_radius, rng_state);
        let volume = (4.0 / 3.0) * std::f32::consts::PI * piece_radius.powi(3);

        pieces.push(DebrisPiece {
            center: piece_center,
            radius: piece_radius,
            mesh,
            volume,
        });

        rng_state = lcg_next(rng_state);
    }

    pieces
}

/// Generate a small convex mesh for a debris piece
///
/// Creates a distorted octahedron centered at the given position.
fn generate_debris_mesh(center: Vec3, radius: f32, seed: u64) -> Mesh {
    let mut rng = seed;

    // 6 base vertices of an octahedron, distorted
    let mut verts = Vec::with_capacity(6);
    let base_dirs = [
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(-1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, -1.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        Vec3::new(0.0, 0.0, -1.0),
    ];

    for dir in &base_dirs {
        rng = lcg_next(rng);
        let distort = 0.7 + lcg_float(rng) * 0.6; // 0.7 to 1.3
        let pos = center + *dir * radius * distort;
        verts.push(pos);
    }

    // 8 triangular faces of the octahedron
    let face_indices: [[usize; 3]; 8] = [
        [0, 2, 4],
        [0, 4, 3],
        [0, 3, 5],
        [0, 5, 2],
        [1, 4, 2],
        [1, 3, 4],
        [1, 5, 3],
        [1, 2, 5],
    ];

    let mut vertices = Vec::with_capacity(24); // 8 faces * 3 verts
    let mut indices = Vec::with_capacity(24);

    for face in &face_indices {
        let a = verts[face[0]];
        let b = verts[face[1]];
        let c = verts[face[2]];
        let normal = (b - a).cross(c - a).normalize_or_zero();

        let vi = vertices.len() as u32;
        vertices.push(Vertex {
            position: a,
            normal,
            ..Default::default()
        });
        vertices.push(Vertex {
            position: b,
            normal,
            ..Default::default()
        });
        vertices.push(Vertex {
            position: c,
            normal,
            ..Default::default()
        });
        indices.push(vi);
        indices.push(vi + 1);
        indices.push(vi + 2);
    }

    Mesh { vertices, indices }
}

/// Simple LCG pseudo-random number generator
#[inline]
fn lcg_next(state: u64) -> u64 {
    state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407)
}

/// Convert LCG state to float in [0, 1)
#[inline]
fn lcg_float(state: u64) -> f32 {
    ((state >> 16) as u32 as f32) / (u32::MAX as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_debris() {
        let pieces = generate_debris(Vec3::ZERO, 1.0, &DebrisConfig::default());

        assert!(!pieces.is_empty());
        for piece in &pieces {
            assert!(piece.mesh.vertices.len() > 0);
            assert!(piece.mesh.indices.len() > 0);
            assert!(piece.radius > 0.0);
            assert!(piece.volume > 0.0);
        }
    }

    #[test]
    fn test_debris_mesh_valid() {
        let mesh = generate_debris_mesh(Vec3::ZERO, 0.5, 42);

        // Octahedron: 8 faces * 3 verts = 24 vertices
        assert_eq!(mesh.vertices.len(), 24);
        assert_eq!(mesh.indices.len(), 24);

        // All indices should be valid
        for &idx in &mesh.indices {
            assert!((idx as usize) < mesh.vertices.len());
        }
    }

    #[test]
    fn test_debris_deterministic() {
        let a = generate_debris(
            Vec3::ZERO,
            1.0,
            &DebrisConfig {
                seed: 123,
                ..Default::default()
            },
        );
        let b = generate_debris(
            Vec3::ZERO,
            1.0,
            &DebrisConfig {
                seed: 123,
                ..Default::default()
            },
        );

        assert_eq!(a.len(), b.len());
        for (pa, pb) in a.iter().zip(b.iter()) {
            assert_eq!(pa.center, pb.center);
            assert_eq!(pa.radius, pb.radius);
        }
    }

    #[test]
    fn test_debris_max_pieces() {
        let config = DebrisConfig {
            max_pieces: 3,
            ..Default::default()
        };
        let pieces = generate_debris(Vec3::ZERO, 1.0, &config);
        assert!(pieces.len() <= 3);
    }
}

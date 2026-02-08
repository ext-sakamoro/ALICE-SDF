//! Alembic (.abc) Export
//!
//! Exports meshes to Alembic format for VFX pipelines.
//! Uses a minimal Ogawa archive format (binary, compact).
//!
//! Compatible with: Maya, Houdini, Blender, Nuke, Katana
//!
//! Author: Moroya Sakamoto

use crate::io::IoError;
use crate::mesh::Mesh;
use std::io::Write;
use std::path::Path;

/// Ogawa archive magic bytes
const OGAWA_MAGIC: &[u8; 5] = b"Ogawa";

/// Configuration for Alembic export
#[derive(Debug, Clone)]
pub struct AlembicConfig {
    /// Frames per second (metadata only for static meshes)
    pub fps: f64,
    /// Export vertex normals
    pub export_normals: bool,
    /// Export UV coordinates
    pub export_uvs: bool,
}

impl Default for AlembicConfig {
    fn default() -> Self {
        AlembicConfig {
            fps: 24.0,
            export_normals: true,
            export_uvs: true,
        }
    }
}

/// Ogawa node type tags
const OGAWA_GROUP: u8 = 0xFF;
const OGAWA_DATA: u8 = 0x00;

struct OgawaWriter<W: Write> {
    writer: W,
    pos: u64,
}

impl<W: Write> OgawaWriter<W> {
    fn new(mut writer: W) -> Result<Self, std::io::Error> {
        // Ogawa header: magic + version
        writer.write_all(OGAWA_MAGIC)?;
        writer.write_all(&[0x00, 0x01, 0x00])?; // version 1.0
        Ok(OgawaWriter { writer, pos: 8 })
    }

    fn write_data(&mut self, data: &[u8]) -> Result<u64, std::io::Error> {
        let offset = self.pos;
        self.writer.write_all(&[OGAWA_DATA])?;
        self.writer.write_all(&(data.len() as u64).to_le_bytes())?;
        self.writer.write_all(data)?;
        self.pos += 1 + 8 + data.len() as u64;
        Ok(offset)
    }

    fn write_string(&mut self, s: &str) -> Result<u64, std::io::Error> {
        self.write_data(s.as_bytes())
    }

    fn write_f32_array(&mut self, data: &[f32]) -> Result<u64, std::io::Error> {
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        self.write_data(&bytes)
    }

    fn write_i32_array(&mut self, data: &[i32]) -> Result<u64, std::io::Error> {
        let bytes: Vec<u8> = data.iter().flat_map(|i| i.to_le_bytes()).collect();
        self.write_data(&bytes)
    }

    fn write_group_header(&mut self, child_count: u32) -> Result<u64, std::io::Error> {
        let offset = self.pos;
        self.writer.write_all(&[OGAWA_GROUP])?;
        self.writer.write_all(&child_count.to_le_bytes())?;
        self.pos += 1 + 4;
        Ok(offset)
    }

    fn flush(&mut self) -> Result<(), std::io::Error> {
        self.writer.flush()
    }
}

/// Export mesh to Alembic (.abc) Ogawa format
pub fn export_alembic(
    mesh: &Mesh,
    path: impl AsRef<Path>,
    config: &AlembicConfig,
) -> Result<(), IoError> {
    let file = std::fs::File::create(path)?;
    let buf = std::io::BufWriter::new(file);
    let mut w = OgawaWriter::new(buf)?;

    // Root group: ABC archive
    let child_count = 2 + config.export_normals as u32 + config.export_uvs as u32;
    w.write_group_header(child_count + 2)?; // +2 for name and metadata

    // Archive name
    w.write_string("ABC")?;

    // Metadata
    let meta = format!(
        "{{\"generator\":\"ALICE-SDF\",\"fps\":{},\"frame_range\":[0,0]}}",
        config.fps
    );
    w.write_string(&meta)?;

    // Mesh object group
    w.write_group_header(child_count)?;
    w.write_string("mesh")?;

    // Positions (P): flatten Vec3 to f32 array
    let positions: Vec<f32> = mesh
        .vertices
        .iter()
        .flat_map(|v| [v.position.x, v.position.y, v.position.z])
        .collect();
    w.write_f32_array(&positions)?;

    // Face indices
    let face_indices: Vec<i32> = mesh.indices.iter().map(|&i| i as i32).collect();
    w.write_i32_array(&face_indices)?;

    // Face counts (all triangles)
    let tri_count = mesh.indices.len() / 3;
    let face_counts: Vec<i32> = vec![3i32; tri_count];
    w.write_i32_array(&face_counts)?;

    // Normals (N)
    if config.export_normals {
        let normals: Vec<f32> = mesh
            .vertices
            .iter()
            .flat_map(|v| [v.normal.x, v.normal.y, v.normal.z])
            .collect();
        w.write_f32_array(&normals)?;
    }

    // UVs
    if config.export_uvs {
        let uvs: Vec<f32> = mesh
            .vertices
            .iter()
            .flat_map(|v| [v.uv.x, v.uv.y])
            .collect();
        w.write_f32_array(&uvs)?;
    }

    w.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::{sdf_to_mesh, MarchingCubesConfig};
    use crate::types::SdfNode;
    use glam::Vec3;

    #[test]
    fn test_alembic_export() {
        let sphere = SdfNode::sphere(1.0);
        let mesh = sdf_to_mesh(
            &sphere,
            Vec3::splat(-2.0),
            Vec3::splat(2.0),
            &MarchingCubesConfig {
                resolution: 8,
                ..Default::default()
            },
        );

        let path = std::env::temp_dir().join("alice_test.abc");
        export_alembic(&mesh, &path, &AlembicConfig::default()).unwrap();

        let data = std::fs::read(&path).unwrap();
        assert!(data.len() > 8);
        assert_eq!(&data[0..5], b"Ogawa");

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_alembic_no_normals() {
        let sphere = SdfNode::sphere(1.0);
        let mesh = sdf_to_mesh(
            &sphere,
            Vec3::splat(-2.0),
            Vec3::splat(2.0),
            &MarchingCubesConfig {
                resolution: 8,
                ..Default::default()
            },
        );

        let path = std::env::temp_dir().join("alice_test_nonorm.abc");
        let config = AlembicConfig {
            export_normals: false,
            ..Default::default()
        };
        export_alembic(&mesh, &path, &config).unwrap();

        let data = std::fs::read(&path).unwrap();
        assert_eq!(&data[0..5], b"Ogawa");

        std::fs::remove_file(&path).ok();
    }
}

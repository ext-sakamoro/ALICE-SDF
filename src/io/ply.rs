//! PLY (Polygon File Format) import/export for ALICE-SDF meshes
//!
//! Supports ASCII and binary little-endian PLY formats.
//! Compatible with MeshLab, CloudCompare, Open3D, and all major DCC tools.
//!
//! Author: Moroya Sakamoto

use crate::io::IoError;
use crate::mesh::Mesh;
use std::path::Path;

/// PLY export configuration
#[derive(Debug, Clone)]
pub struct PlyConfig {
    /// Export as binary (true) or ASCII (false)
    pub binary: bool,
    /// Export vertex normals
    pub export_normals: bool,
}

impl Default for PlyConfig {
    fn default() -> Self {
        PlyConfig {
            binary: true,
            export_normals: true,
        }
    }
}

/// Export mesh to PLY format
pub fn export_ply(
    mesh: &Mesh,
    path: impl AsRef<Path>,
    config: &PlyConfig,
) -> Result<(), IoError> {
    use std::io::{BufWriter, Write};

    let file = std::fs::File::create(path)?;
    let mut w = BufWriter::new(file);

    let vert_count = mesh.vertices.len();
    let face_count = mesh.indices.len() / 3;
    let format = if config.binary {
        "binary_little_endian 1.0"
    } else {
        "ascii 1.0"
    };

    // Header
    writeln!(w, "ply")?;
    writeln!(w, "format {}", format)?;
    writeln!(w, "comment ALICE-SDF PLY Export")?;
    writeln!(w, "element vertex {}", vert_count)?;
    writeln!(w, "property float x")?;
    writeln!(w, "property float y")?;
    writeln!(w, "property float z")?;
    if config.export_normals {
        writeln!(w, "property float nx")?;
        writeln!(w, "property float ny")?;
        writeln!(w, "property float nz")?;
    }
    writeln!(w, "element face {}", face_count)?;
    writeln!(w, "property list uchar int vertex_indices")?;
    writeln!(w, "end_header")?;

    if config.binary {
        // Binary vertex data
        for v in &mesh.vertices {
            w.write_all(&v.position.x.to_le_bytes())?;
            w.write_all(&v.position.y.to_le_bytes())?;
            w.write_all(&v.position.z.to_le_bytes())?;
            if config.export_normals {
                w.write_all(&v.normal.x.to_le_bytes())?;
                w.write_all(&v.normal.y.to_le_bytes())?;
                w.write_all(&v.normal.z.to_le_bytes())?;
            }
        }
        // Binary face data
        for i in 0..face_count {
            w.write_all(&[3u8])?;
            let i0 = mesh.indices[i * 3] as i32;
            let i1 = mesh.indices[i * 3 + 1] as i32;
            let i2 = mesh.indices[i * 3 + 2] as i32;
            w.write_all(&i0.to_le_bytes())?;
            w.write_all(&i1.to_le_bytes())?;
            w.write_all(&i2.to_le_bytes())?;
        }
    } else {
        // ASCII vertex data
        for v in &mesh.vertices {
            if config.export_normals {
                writeln!(
                    w,
                    "{} {} {} {} {} {}",
                    v.position.x, v.position.y, v.position.z,
                    v.normal.x, v.normal.y, v.normal.z
                )?;
            } else {
                writeln!(w, "{} {} {}", v.position.x, v.position.y, v.position.z)?;
            }
        }
        // ASCII face data
        for i in 0..face_count {
            writeln!(
                w,
                "3 {} {} {}",
                mesh.indices[i * 3],
                mesh.indices[i * 3 + 1],
                mesh.indices[i * 3 + 2]
            )?;
        }
    }

    w.flush()?;
    Ok(())
}

/// Import mesh from PLY format (auto-detects ASCII vs binary)
pub fn import_ply(path: impl AsRef<Path>) -> Result<Mesh, IoError> {
    use crate::mesh::Vertex;
    use glam::Vec3;

    let data = std::fs::read(path)?;

    // Find end_header marker by scanning bytes
    let marker = b"end_header\n";
    let header_end = data
        .windows(marker.len())
        .position(|w| w == marker)
        .map(|pos| pos + marker.len())
        .ok_or_else(|| IoError::InvalidFormat("Missing end_header in PLY".into()))?;

    let header_text = std::str::from_utf8(&data[..header_end])
        .map_err(|e| IoError::InvalidFormat(format!("Invalid header: {}", e)))?;

    // Parse header
    let mut is_binary = false;
    let mut vert_count: usize = 0;
    let mut face_count: usize = 0;
    let mut has_normals = false;
    let mut in_vertex_elem = false;

    for line in header_text.lines() {
        let line = line.trim();
        if line.starts_with("format binary_little_endian") {
            is_binary = true;
        }
        if line.starts_with("element vertex") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            vert_count = parts.get(2).and_then(|s| s.parse().ok()).unwrap_or(0);
            in_vertex_elem = true;
        }
        if line.starts_with("element face") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            face_count = parts.get(2).and_then(|s| s.parse().ok()).unwrap_or(0);
            in_vertex_elem = false;
        }
        if in_vertex_elem && line == "property float nx" {
            has_normals = true;
        }
    }

    if is_binary {
        let floats_per_vert = if has_normals { 6 } else { 3 };
        let vert_bytes = vert_count * floats_per_vert * 4;
        let face_bytes = face_count * (1 + 3 * 4); // 1 byte count + 3 x i32

        if data.len() < header_end + vert_bytes + face_bytes {
            return Err(IoError::InvalidFormat("PLY binary data truncated".into()));
        }

        let mut vertices = Vec::with_capacity(vert_count);
        let mut offset = header_end;

        let read_f32 = |o: usize| -> f32 {
            f32::from_le_bytes([data[o], data[o + 1], data[o + 2], data[o + 3]])
        };

        for _ in 0..vert_count {
            let x = read_f32(offset);
            let y = read_f32(offset + 4);
            let z = read_f32(offset + 8);
            let normal = if has_normals {
                let nx = read_f32(offset + 12);
                let ny = read_f32(offset + 16);
                let nz = read_f32(offset + 20);
                Vec3::new(nx, ny, nz)
            } else {
                Vec3::Y
            };
            vertices.push(Vertex::new(Vec3::new(x, y, z), normal));
            offset += floats_per_vert * 4;
        }

        let mut indices = Vec::with_capacity(face_count * 3);
        for _ in 0..face_count {
            let _count = data[offset]; // should be 3
            offset += 1;
            for _ in 0..3 {
                let idx = i32::from_le_bytes([
                    data[offset],
                    data[offset + 1],
                    data[offset + 2],
                    data[offset + 3],
                ]) as u32;
                indices.push(idx);
                offset += 4;
            }
        }

        Ok(Mesh { vertices, indices })
    } else {
        // ASCII
        let text = std::str::from_utf8(&data)
            .map_err(|e| IoError::InvalidFormat(format!("Invalid UTF-8: {}", e)))?;

        let mut vertices = Vec::with_capacity(vert_count);
        let mut indices = Vec::with_capacity(face_count * 3);
        let mut past_header = false;
        let mut verts_read = 0usize;

        for line in text.lines() {
            let line = line.trim();
            if line == "end_header" {
                past_header = true;
                continue;
            }
            if !past_header {
                continue;
            }

            let parts: Vec<&str> = line.split_whitespace().collect();
            if verts_read < vert_count {
                if parts.len() >= 3 {
                    let x: f32 = parts[0].parse().unwrap_or(0.0);
                    let y: f32 = parts[1].parse().unwrap_or(0.0);
                    let z: f32 = parts[2].parse().unwrap_or(0.0);
                    let normal = if has_normals && parts.len() >= 6 {
                        let nx: f32 = parts[3].parse().unwrap_or(0.0);
                        let ny: f32 = parts[4].parse().unwrap_or(0.0);
                        let nz: f32 = parts[5].parse().unwrap_or(0.0);
                        Vec3::new(nx, ny, nz)
                    } else {
                        Vec3::Y
                    };
                    vertices.push(Vertex::new(Vec3::new(x, y, z), normal));
                }
                verts_read += 1;
            } else if parts.len() >= 4 {
                // Face: "3 i0 i1 i2"
                let i0: u32 = parts[1].parse().unwrap_or(0);
                let i1: u32 = parts[2].parse().unwrap_or(0);
                let i2: u32 = parts[3].parse().unwrap_or(0);
                indices.push(i0);
                indices.push(i1);
                indices.push(i2);
            }
        }

        Ok(Mesh { vertices, indices })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::{sdf_to_mesh, MarchingCubesConfig};
    use crate::types::SdfNode;
    use glam::Vec3;

    #[test]
    fn test_ply_binary_export_import_roundtrip() {
        let sphere = SdfNode::sphere(1.0);
        let config = MarchingCubesConfig {
            resolution: 8,
            iso_level: 0.0,
            compute_normals: true,
            ..Default::default()
        };
        let mesh = sdf_to_mesh(&sphere, Vec3::splat(-2.0), Vec3::splat(2.0), &config);
        let path = std::env::temp_dir().join("alice_test_ply_bin.ply");
        export_ply(&mesh, &path, &PlyConfig::default()).unwrap();
        let imported = import_ply(&path).unwrap();
        assert!(imported.vertex_count() > 0);
        assert_eq!(imported.triangle_count(), mesh.triangle_count());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_ply_ascii_export_import_roundtrip() {
        let sphere = SdfNode::sphere(1.0);
        let config = MarchingCubesConfig {
            resolution: 4,
            iso_level: 0.0,
            compute_normals: true,
            ..Default::default()
        };
        let mesh = sdf_to_mesh(&sphere, Vec3::splat(-2.0), Vec3::splat(2.0), &config);
        let path = std::env::temp_dir().join("alice_test_ply_ascii.ply");
        let ply_config = PlyConfig {
            binary: false,
            export_normals: true,
        };
        export_ply(&mesh, &path, &ply_config).unwrap();
        let imported = import_ply(&path).unwrap();
        assert!(imported.vertex_count() > 0);
        assert_eq!(imported.triangle_count(), mesh.triangle_count());
        std::fs::remove_file(&path).ok();
    }
}

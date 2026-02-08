//! STL (Stereolithography) import/export for ALICE-SDF meshes
//!
//! Supports both binary and ASCII STL formats.
//! Compatible with 3D printing software and all major DCC tools.
//!
//! Author: Moroya Sakamoto

use crate::io::IoError;
use crate::mesh::Mesh;
use std::path::Path;

/// Export mesh to binary STL format
pub fn export_stl(mesh: &Mesh, path: impl AsRef<Path>) -> Result<(), IoError> {
    use std::io::{BufWriter, Write};

    let file = std::fs::File::create(path)?;
    let mut w = BufWriter::new(file);

    // 80-byte header
    let header = [0u8; 80];
    w.write_all(&header)?;

    // Triangle count
    let tri_count = (mesh.indices.len() / 3) as u32;
    w.write_all(&tri_count.to_le_bytes())?;

    // Each triangle: normal(3xf32) + v1(3xf32) + v2(3xf32) + v3(3xf32) + u16 attr
    for i in 0..tri_count as usize {
        let i0 = mesh.indices[i * 3] as usize;
        let i1 = mesh.indices[i * 3 + 1] as usize;
        let i2 = mesh.indices[i * 3 + 2] as usize;

        let v0 = &mesh.vertices[i0];
        let v1 = &mesh.vertices[i1];
        let v2 = &mesh.vertices[i2];

        // Face normal (average of vertex normals)
        let n = (v0.normal + v1.normal + v2.normal).normalize_or_zero();
        for f in [n.x, n.y, n.z] {
            w.write_all(&f.to_le_bytes())?;
        }
        for v in [v0, v1, v2] {
            for f in [v.position.x, v.position.y, v.position.z] {
                w.write_all(&f.to_le_bytes())?;
            }
        }
        w.write_all(&0u16.to_le_bytes())?;
    }

    w.flush()?;
    Ok(())
}

/// Export mesh to ASCII STL format
pub fn export_stl_ascii(mesh: &Mesh, path: impl AsRef<Path>) -> Result<(), IoError> {
    use std::io::{BufWriter, Write};

    let file = std::fs::File::create(path)?;
    let mut w = BufWriter::new(file);

    writeln!(w, "solid alice_sdf")?;

    let tri_count = mesh.indices.len() / 3;
    for i in 0..tri_count {
        let i0 = mesh.indices[i * 3] as usize;
        let i1 = mesh.indices[i * 3 + 1] as usize;
        let i2 = mesh.indices[i * 3 + 2] as usize;

        let v0 = &mesh.vertices[i0];
        let v1 = &mesh.vertices[i1];
        let v2 = &mesh.vertices[i2];

        let n = (v0.normal + v1.normal + v2.normal).normalize_or_zero();

        writeln!(w, "  facet normal {} {} {}", n.x, n.y, n.z)?;
        writeln!(w, "    outer loop")?;
        for v in [v0, v1, v2] {
            writeln!(
                w,
                "      vertex {} {} {}",
                v.position.x, v.position.y, v.position.z
            )?;
        }
        writeln!(w, "    endloop")?;
        writeln!(w, "  endfacet")?;
    }

    writeln!(w, "endsolid alice_sdf")?;
    w.flush()?;
    Ok(())
}

/// Import mesh from STL format (auto-detects binary vs ASCII)
pub fn import_stl(path: impl AsRef<Path>) -> Result<Mesh, IoError> {
    let data = std::fs::read(path)?;
    if data.len() < 84 {
        return Err(IoError::InvalidFormat("STL file too small".into()));
    }

    // Heuristic: if starts with "solid " and contains "facet", treat as ASCII
    let is_ascii = data.starts_with(b"solid ")
        && std::str::from_utf8(&data[..data.len().min(1000)])
            .map(|s| s.contains("facet"))
            .unwrap_or(false);

    if is_ascii {
        import_stl_ascii(&data)
    } else {
        import_stl_binary(&data)
    }
}

fn import_stl_binary(data: &[u8]) -> Result<Mesh, IoError> {
    use crate::mesh::Vertex;
    use glam::Vec3;

    if data.len() < 84 {
        return Err(IoError::InvalidFormat("Binary STL too small".into()));
    }

    let tri_count = u32::from_le_bytes([data[80], data[81], data[82], data[83]]) as usize;
    let expected = 84 + tri_count * 50;
    if data.len() < expected {
        return Err(IoError::InvalidFormat(format!(
            "Binary STL truncated: expected {} bytes, got {}",
            expected,
            data.len()
        )));
    }

    let mut vertices = Vec::with_capacity(tri_count * 3);
    let mut indices = Vec::with_capacity(tri_count * 3);
    let mut vertex_map: std::collections::HashMap<[u32; 3], u32> = std::collections::HashMap::new();

    let read_f32 = |offset: usize| -> f32 {
        f32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ])
    };

    for t in 0..tri_count {
        let base = 84 + t * 50;
        let nx = read_f32(base);
        let ny = read_f32(base + 4);
        let nz = read_f32(base + 8);
        let normal = Vec3::new(nx, ny, nz);

        for v in 0..3 {
            let vb = base + 12 + v * 12;
            let px = read_f32(vb);
            let py = read_f32(vb + 4);
            let pz = read_f32(vb + 8);

            let key = [px.to_bits(), py.to_bits(), pz.to_bits()];
            let idx = if let Some(&existing) = vertex_map.get(&key) {
                existing
            } else {
                let idx = vertices.len() as u32;
                vertices.push(Vertex::new(Vec3::new(px, py, pz), normal));
                vertex_map.insert(key, idx);
                idx
            };
            indices.push(idx);
        }
    }

    Ok(Mesh { vertices, indices })
}

fn import_stl_ascii(data: &[u8]) -> Result<Mesh, IoError> {
    use crate::mesh::Vertex;
    use glam::Vec3;

    let text = std::str::from_utf8(data)
        .map_err(|e| IoError::InvalidFormat(format!("Invalid UTF-8: {}", e)))?;

    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let mut vertex_map: std::collections::HashMap<[u32; 3], u32> = std::collections::HashMap::new();
    let mut current_normal = Vec3::Y;

    for line in text.lines() {
        let line = line.trim();
        if line.starts_with("facet normal") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 5 {
                let nx: f32 = parts[2].parse().unwrap_or(0.0);
                let ny: f32 = parts[3].parse().unwrap_or(0.0);
                let nz: f32 = parts[4].parse().unwrap_or(0.0);
                current_normal = Vec3::new(nx, ny, nz);
            }
        } else if line.starts_with("vertex") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 4 {
                let px: f32 = parts[1].parse().unwrap_or(0.0);
                let py: f32 = parts[2].parse().unwrap_or(0.0);
                let pz: f32 = parts[3].parse().unwrap_or(0.0);

                let key = [px.to_bits(), py.to_bits(), pz.to_bits()];
                let idx = if let Some(&existing) = vertex_map.get(&key) {
                    existing
                } else {
                    let idx = vertices.len() as u32;
                    vertices.push(Vertex::new(Vec3::new(px, py, pz), current_normal));
                    vertex_map.insert(key, idx);
                    idx
                };
                indices.push(idx);
            }
        }
    }

    Ok(Mesh { vertices, indices })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::{sdf_to_mesh, MarchingCubesConfig};
    use crate::types::SdfNode;
    use glam::Vec3;

    #[test]
    fn test_stl_binary_export_import_roundtrip() {
        let sphere = SdfNode::sphere(1.0);
        let config = MarchingCubesConfig {
            resolution: 8,
            iso_level: 0.0,
            compute_normals: true,
            ..Default::default()
        };
        let mesh = sdf_to_mesh(&sphere, Vec3::splat(-2.0), Vec3::splat(2.0), &config);
        let path = std::env::temp_dir().join("alice_test_stl_bin.stl");
        export_stl(&mesh, &path).unwrap();
        let imported = import_stl(&path).unwrap();
        assert!(imported.vertex_count() > 0);
        assert_eq!(imported.triangle_count(), mesh.triangle_count());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_stl_ascii_export_import_roundtrip() {
        let sphere = SdfNode::sphere(1.0);
        let config = MarchingCubesConfig {
            resolution: 4,
            iso_level: 0.0,
            compute_normals: true,
            ..Default::default()
        };
        let mesh = sdf_to_mesh(&sphere, Vec3::splat(-2.0), Vec3::splat(2.0), &config);
        let path = std::env::temp_dir().join("alice_test_stl_ascii.stl");
        export_stl_ascii(&mesh, &path).unwrap();
        let imported = import_stl(&path).unwrap();
        assert!(imported.vertex_count() > 0);
        assert_eq!(imported.triangle_count(), mesh.triangle_count());
        std::fs::remove_file(&path).ok();
    }
}

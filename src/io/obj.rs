//! Wavefront OBJ export for ALICE-SDF meshes
//!
//! Exports meshes to standard .obj format with optional .mtl material files.
//! Compatible with Blender, Maya, 3ds Max, and all major DCC tools.
//!
//! Author: Moroya Sakamoto

use crate::io::IoError;
use crate::material::MaterialLibrary;
use crate::mesh::Mesh;
use std::io::Write;
use std::path::Path;

/// OBJ export configuration
#[derive(Debug, Clone)]
pub struct ObjConfig {
    /// Export normals (vn)
    pub export_normals: bool,
    /// Export UVs (vt)
    pub export_uvs: bool,
    /// Export materials (.mtl)
    pub export_materials: bool,
    /// Flip UV V coordinate (some engines use 1-v)
    pub flip_uv_v: bool,
}

impl Default for ObjConfig {
    fn default() -> Self {
        ObjConfig {
            export_normals: true,
            export_uvs: true,
            export_materials: true,
            flip_uv_v: false,
        }
    }
}

/// Export a mesh to Wavefront OBJ format
pub fn export_obj(
    mesh: &Mesh,
    path: impl AsRef<Path>,
    config: &ObjConfig,
    materials: Option<&MaterialLibrary>,
) -> Result<(), IoError> {
    let path = path.as_ref();
    let file = std::fs::File::create(path)?;
    let mut w = std::io::BufWriter::new(file);

    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("mesh");

    // Header
    writeln!(w, "# ALICE-SDF OBJ Export")?;
    writeln!(w, "# Vertices: {}", mesh.vertices.len())?;
    writeln!(w, "# Triangles: {}", mesh.indices.len() / 3)?;

    // MTL reference
    if config.export_materials && materials.is_some() {
        writeln!(w, "mtllib {}.mtl", stem)?;
    }

    writeln!(w, "o {}", stem)?;

    // Positions
    for v in &mesh.vertices {
        writeln!(w, "v {} {} {}", v.position.x, v.position.y, v.position.z)?;
    }

    // UVs
    if config.export_uvs {
        for v in &mesh.vertices {
            let v_coord = if config.flip_uv_v { 1.0 - v.uv.y } else { v.uv.y };
            writeln!(w, "vt {} {}", v.uv.x, v_coord)?;
        }
    }

    // Normals
    if config.export_normals {
        for v in &mesh.vertices {
            writeln!(w, "vn {} {} {}", v.normal.x, v.normal.y, v.normal.z)?;
        }
    }

    // Faces - group by material_id
    let tri_count = mesh.indices.len() / 3;

    if config.export_materials && materials.is_some() {
        let mat_lib = materials.unwrap();
        // Collect triangles by material
        let mut mat_tris: std::collections::BTreeMap<u32, Vec<usize>> =
            std::collections::BTreeMap::new();
        for i in 0..tri_count {
            let mat_id = mesh.vertices[mesh.indices[i * 3] as usize].material_id;
            mat_tris.entry(mat_id).or_default().push(i);
        }

        for (mat_id, tris) in &mat_tris {
            let mat_name = if (*mat_id as usize) < mat_lib.materials.len() {
                &mat_lib.materials[*mat_id as usize].name
            } else {
                "default"
            };
            writeln!(w, "usemtl {}", mat_name)?;

            for &i in tris {
                write_face(&mut w, mesh, i, config)?;
            }
        }
    } else {
        for i in 0..tri_count {
            write_face(&mut w, mesh, i, config)?;
        }
    }

    w.flush()?;

    // Export MTL file
    if config.export_materials {
        if let Some(mat_lib) = materials {
            let mtl_path = path.with_extension("mtl");
            export_mtl(mat_lib, &mtl_path)?;
        }
    }

    Ok(())
}

fn write_face(
    w: &mut impl Write,
    mesh: &Mesh,
    tri_idx: usize,
    config: &ObjConfig,
) -> Result<(), IoError> {
    let base = tri_idx * 3;
    let a = mesh.indices[base] as usize + 1; // OBJ is 1-indexed
    let b = mesh.indices[base + 1] as usize + 1;
    let c = mesh.indices[base + 2] as usize + 1;

    if config.export_normals && config.export_uvs {
        writeln!(w, "f {}/{}/{} {}/{}/{} {}/{}/{}", a, a, a, b, b, b, c, c, c)?;
    } else if config.export_normals {
        writeln!(w, "f {}//{} {}//{} {}//{}", a, a, b, b, c, c)?;
    } else if config.export_uvs {
        writeln!(w, "f {}/{} {}/{} {}/{}", a, a, b, b, c, c)?;
    } else {
        writeln!(w, "f {} {} {}", a, b, c)?;
    }

    Ok(())
}

/// Export materials to MTL format
fn export_mtl(mat_lib: &MaterialLibrary, path: impl AsRef<Path>) -> Result<(), IoError> {
    let file = std::fs::File::create(path)?;
    let mut w = std::io::BufWriter::new(file);

    writeln!(w, "# ALICE-SDF MTL Export")?;

    for mat in &mat_lib.materials {
        writeln!(w, "\nnewmtl {}", mat.name)?;

        // Diffuse (Kd)
        writeln!(
            w,
            "Kd {} {} {}",
            mat.base_color[0], mat.base_color[1], mat.base_color[2]
        )?;

        // Ambient (Ka) - use diffuse * 0.1
        writeln!(
            w,
            "Ka {} {} {}",
            mat.base_color[0] * 0.1,
            mat.base_color[1] * 0.1,
            mat.base_color[2] * 0.1
        )?;

        // Specular (Ks) - from metallic
        let spec = mat.metallic;
        writeln!(w, "Ks {} {} {}", spec, spec, spec)?;

        // Specular exponent (Ns) - from roughness
        let ns = (1.0 - mat.roughness) * 900.0 + 10.0;
        writeln!(w, "Ns {}", ns)?;

        // Opacity
        writeln!(w, "d {}", mat.opacity)?;

        // Illumination model
        if mat.metallic > 0.5 {
            writeln!(w, "illum 3")?; // Metallic
        } else {
            writeln!(w, "illum 2")?; // Dielectric
        }

        // Emission (Ke) - PBR extension
        if mat.emission_strength > 0.0 {
            writeln!(
                w,
                "Ke {} {} {}",
                mat.emission[0] * mat.emission_strength,
                mat.emission[1] * mat.emission_strength,
                mat.emission[2] * mat.emission_strength,
            )?;
        }

        // IOR (Ni)
        writeln!(w, "Ni {}", mat.ior)?;

        // Texture maps
        if let Some(ref tex) = mat.albedo_map {
            writeln!(w, "map_Kd {}", tex.path)?;
        }
        if let Some(ref tex) = mat.normal_map {
            writeln!(w, "bump {}", tex.path)?;
        }
        if let Some(ref tex) = mat.metallic_map {
            writeln!(w, "map_Ks {}", tex.path)?;
        }
        if let Some(ref tex) = mat.roughness_map {
            writeln!(w, "map_Ns {}", tex.path)?;
        }
        if let Some(ref tex) = mat.ao_map {
            writeln!(w, "map_Ka {}", tex.path)?;
        }
        if let Some(ref tex) = mat.emissive_map {
            writeln!(w, "map_Ke {}", tex.path)?;
        }
    }

    w.flush()?;
    Ok(())
}

/// Load a mesh from OBJ format
pub fn import_obj(path: impl AsRef<Path>) -> Result<Mesh, IoError> {
    use crate::mesh::Vertex;
    use glam::{Vec2, Vec3};

    let content =
        std::fs::read_to_string(path).map_err(IoError::Io)?;

    let mut positions: Vec<Vec3> = Vec::new();
    let mut normals: Vec<Vec3> = Vec::new();
    let mut uvs: Vec<Vec2> = Vec::new();

    let mut vertices: Vec<Vertex> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    // Map (pos_idx, uv_idx, norm_idx) -> vertex index for deduplication
    let mut vertex_map: std::collections::HashMap<(usize, usize, usize), u32> =
        std::collections::HashMap::new();

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() {
            continue;
        }

        match parts[0] {
            "v" if parts.len() >= 4 => {
                let x: f32 = parts[1].parse().unwrap_or(0.0);
                let y: f32 = parts[2].parse().unwrap_or(0.0);
                let z: f32 = parts[3].parse().unwrap_or(0.0);
                positions.push(Vec3::new(x, y, z));
            }
            "vn" if parts.len() >= 4 => {
                let x: f32 = parts[1].parse().unwrap_or(0.0);
                let y: f32 = parts[2].parse().unwrap_or(0.0);
                let z: f32 = parts[3].parse().unwrap_or(0.0);
                normals.push(Vec3::new(x, y, z).normalize());
            }
            "vt" if parts.len() >= 3 => {
                let u: f32 = parts[1].parse().unwrap_or(0.0);
                let v: f32 = parts[2].parse().unwrap_or(0.0);
                uvs.push(Vec2::new(u, v));
            }
            "f" if parts.len() >= 4 => {
                // Triangulate face (fan triangulation for n-gons)
                let face_verts: Vec<u32> = parts[1..]
                    .iter()
                    .filter_map(|&s| {
                        let (pi, ui, ni) = parse_face_vertex(s);
                        let key = (pi, ui, ni);

                        if let Some(&idx) = vertex_map.get(&key) {
                            Some(idx)
                        } else {
                            let pos = if pi > 0 && pi <= positions.len() {
                                positions[pi - 1]
                            } else {
                                Vec3::ZERO
                            };
                            let norm = if ni > 0 && ni <= normals.len() {
                                normals[ni - 1]
                            } else {
                                Vec3::Y
                            };
                            let uv = if ui > 0 && ui <= uvs.len() {
                                uvs[ui - 1]
                            } else {
                                Vec2::ZERO
                            };

                            let idx = vertices.len() as u32;
                            let mut vert = Vertex::new(pos, norm);
                            vert.uv = uv;
                            vertices.push(vert);
                            vertex_map.insert(key, idx);
                            Some(idx)
                        }
                    })
                    .collect();

                // Fan triangulation
                for i in 1..face_verts.len().saturating_sub(1) {
                    indices.push(face_verts[0]);
                    indices.push(face_verts[i]);
                    indices.push(face_verts[i + 1]);
                }
            }
            _ => {}
        }
    }

    Ok(Mesh { vertices, indices })
}

/// Parse OBJ face vertex: v, v/vt, v/vt/vn, v//vn
fn parse_face_vertex(s: &str) -> (usize, usize, usize) {
    let parts: Vec<&str> = s.split('/').collect();
    let v = parts.first().and_then(|s| s.parse().ok()).unwrap_or(0);
    let vt = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);
    let vn = parts.get(2).and_then(|s| s.parse().ok()).unwrap_or(0);
    (v, vt, vn)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::{sdf_to_mesh, MarchingCubesConfig};
    use crate::types::SdfNode;
    use glam::Vec3;

    #[test]
    fn test_obj_export_import_roundtrip() {
        let sphere = SdfNode::sphere(1.0);
        let config = MarchingCubesConfig {
            resolution: 8,
            iso_level: 0.0,
            compute_normals: true,
            ..Default::default()
        };
        let mesh = sdf_to_mesh(&sphere, Vec3::splat(-2.0), Vec3::splat(2.0), &config);

        let path = std::env::temp_dir().join("alice_test_export.obj");
        export_obj(&mesh, &path, &ObjConfig::default(), None).unwrap();

        let imported = import_obj(&path).unwrap();
        assert!(imported.vertex_count() > 0);
        assert!(imported.triangle_count() > 0);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_obj_with_materials() {
        let sphere = SdfNode::sphere(1.0);
        let config = MarchingCubesConfig {
            resolution: 8,
            iso_level: 0.0,
            compute_normals: true,
            ..Default::default()
        };
        let mesh = sdf_to_mesh(&sphere, Vec3::splat(-2.0), Vec3::splat(2.0), &config);

        let mut mat_lib = MaterialLibrary::new();
        mat_lib.add(crate::material::Material::metal("gold", 1.0, 0.84, 0.0, 0.3));

        let path = std::env::temp_dir().join("alice_test_mat.obj");
        export_obj(&mesh, &path, &ObjConfig::default(), Some(&mat_lib)).unwrap();

        let mtl_path = path.with_extension("mtl");
        assert!(mtl_path.exists());

        std::fs::remove_file(&path).ok();
        std::fs::remove_file(&mtl_path).ok();
    }

    #[test]
    fn test_obj_minimal() {
        let sphere = SdfNode::sphere(1.0);
        let config = MarchingCubesConfig {
            resolution: 4,
            iso_level: 0.0,
            compute_normals: false,
            ..Default::default()
        };
        let mesh = sdf_to_mesh(&sphere, Vec3::splat(-2.0), Vec3::splat(2.0), &config);

        let path = std::env::temp_dir().join("alice_test_minimal.obj");
        let obj_config = ObjConfig {
            export_normals: false,
            export_uvs: false,
            export_materials: false,
            flip_uv_v: false,
        };
        export_obj(&mesh, &path, &obj_config, None).unwrap();

        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("f "));
        assert!(!content.contains("vn "));
        assert!(!content.contains("vt "));

        std::fs::remove_file(&path).ok();
    }
}

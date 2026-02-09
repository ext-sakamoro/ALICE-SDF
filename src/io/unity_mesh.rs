//! Unity mesh asset export (Deep Fried Edition)
//!
//! Generates Unity-compatible mesh data as a JSON asset manifest
//! that can be imported by Unity's ALICE-SDF integration package,
//! plus a compact binary variant for runtime loading.
//!
//! # Formats
//!
//! - `.unity_mesh` (JSON): Human-readable asset manifest with vertex/index data
//! - `.unity_mesh_bin` (Binary): Compact interleaved vertex layout + raw indices
//!
//! # Coordinate Conversion
//!
//! Unity uses a **left-handed** coordinate system with Y-up.
//! ALICE-SDF uses a right-handed coordinate system.
//! - Flip Z: `position.z = -position.z`, `normal.z = -normal.z`
//! - Flip winding: swap `indices[1]` and `indices[2]` per triangle
//!
//! # Deep Fried Optimizations
//! - **Buffered I/O**: Uses `BufWriter` for efficient buffering.
//! - **Pre-allocated buffers**: Flat arrays sized up-front to avoid realloc.
//!
//! Author: Moroya Sakamoto

use crate::io::IoError;
use crate::mesh::Mesh;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Magic bytes for Unity binary mesh format
const UNITY_BIN_MAGIC: &[u8; 4] = b"ALMB";

/// Unity binary mesh version
const UNITY_BIN_VERSION: u16 = 1;

/// JSON format version
const UNITY_JSON_VERSION: u32 = 1;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Unity mesh export configuration
#[derive(Debug, Clone)]
pub struct UnityMeshConfig {
    /// Scale factor (Unity uses meters, SDF might use different units)
    pub scale: f32,
    /// Whether to flip Z axis (Unity is left-handed)
    pub flip_z: bool,
    /// Whether to flip winding order (required for left-handed)
    pub flip_winding: bool,
    /// Asset name
    pub name: String,
}

impl Default for UnityMeshConfig {
    fn default() -> Self {
        Self {
            scale: 1.0,
            flip_z: true,
            flip_winding: true,
            name: "AliceSdfMesh".to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// Coordinate helpers
// ---------------------------------------------------------------------------

/// Apply Unity coordinate conversion to a position or normal vector.
/// Flips Z if configured, applies scale to positions (not normals).
#[inline(always)]
fn convert_position(x: f32, y: f32, z: f32, config: &UnityMeshConfig) -> (f32, f32, f32) {
    let s = config.scale;
    let z_out = if config.flip_z { -z * s } else { z * s };
    (x * s, y * s, z_out)
}

#[inline(always)]
fn convert_normal(x: f32, y: f32, z: f32, config: &UnityMeshConfig) -> (f32, f32, f32) {
    let z_out = if config.flip_z { -z } else { z };
    (x, y, z_out)
}

#[inline(always)]
fn convert_tangent(
    x: f32,
    y: f32,
    z: f32,
    w: f32,
    config: &UnityMeshConfig,
) -> (f32, f32, f32, f32) {
    let z_out = if config.flip_z { -z } else { z };
    // Flip tangent w (handedness) when flipping coordinate system
    let w_out = if config.flip_z { -w } else { w };
    (x, y, z_out, w_out)
}

// ---------------------------------------------------------------------------
// JSON export
// ---------------------------------------------------------------------------

/// Export mesh as Unity-compatible JSON asset (.unity_mesh)
///
/// Generates a JSON manifest containing all vertex data (positions, normals,
/// UVs, tangents, colors) and index data, with coordinate system conversion
/// applied.
///
/// # Arguments
/// * `mesh` - Source mesh
/// * `path` - Destination file path
/// * `config` - Unity export configuration
///
/// # Errors
/// Returns `IoError::Io` on filesystem errors, `IoError::Serialization` on
/// JSON encoding errors.
pub fn export_unity_mesh(
    mesh: &Mesh,
    path: impl AsRef<Path>,
    config: &UnityMeshConfig,
) -> Result<(), IoError> {
    let file = File::create(path)?;
    let mut w = BufWriter::new(file);

    let vc = mesh.vertices.len();
    let ic = mesh.indices.len();

    // Pre-allocate flat arrays
    let mut positions = Vec::with_capacity(vc * 3);
    let mut normals = Vec::with_capacity(vc * 3);
    let mut uvs = Vec::with_capacity(vc * 2);
    let mut tangents = Vec::with_capacity(vc * 4);
    let mut colors = Vec::with_capacity(vc * 4);

    for v in &mesh.vertices {
        let (px, py, pz) = convert_position(
            v.position.x,
            v.position.y,
            v.position.z,
            config,
        );
        positions.push(px);
        positions.push(py);
        positions.push(pz);

        let (nx, ny, nz) = convert_normal(v.normal.x, v.normal.y, v.normal.z, config);
        normals.push(nx);
        normals.push(ny);
        normals.push(nz);

        uvs.push(v.uv.x);
        uvs.push(v.uv.y);

        let (tx, ty, tz, tw) =
            convert_tangent(v.tangent.x, v.tangent.y, v.tangent.z, v.tangent.w, config);
        tangents.push(tx);
        tangents.push(ty);
        tangents.push(tz);
        tangents.push(tw);

        colors.push(v.color[0]);
        colors.push(v.color[1]);
        colors.push(v.color[2]);
        colors.push(v.color[3]);
    }

    // Convert indices (flip winding per triangle)
    let mut indices = Vec::with_capacity(ic);
    let tri_count = ic / 3;
    for t in 0..tri_count {
        let base = t * 3;
        let i0 = mesh.indices[base];
        let i1 = mesh.indices[base + 1];
        let i2 = mesh.indices[base + 2];
        if config.flip_winding {
            indices.push(i0);
            indices.push(i2);
            indices.push(i1);
        } else {
            indices.push(i0);
            indices.push(i1);
            indices.push(i2);
        }
    }

    // Build JSON manually for performance (avoid serde overhead for large arrays)
    write!(w, "{{\n  \"alice_sdf_mesh\": {{\n")?;
    write!(w, "    \"version\": {},\n", UNITY_JSON_VERSION)?;
    write!(w, "    \"name\": \"{}\",\n", config.name)?;
    write!(w, "    \"vertex_count\": {},\n", vc)?;
    write!(w, "    \"index_count\": {},\n", ic)?;

    // Sub-meshes (single sub-mesh for now)
    write!(
        w,
        "    \"sub_meshes\": [{{ \"topology\": 0, \"index_start\": 0, \"index_count\": {} }}],\n",
        ic
    )?;

    // Vertex data
    write!(w, "    \"vertex_data\": {{\n")?;
    write_json_f32_array(&mut w, "positions", &positions, 6)?;
    write!(w, ",\n")?;
    write_json_f32_array(&mut w, "normals", &normals, 6)?;
    write!(w, ",\n")?;
    write_json_f32_array(&mut w, "uvs", &uvs, 6)?;
    write!(w, ",\n")?;
    write_json_f32_array(&mut w, "tangents", &tangents, 6)?;
    write!(w, ",\n")?;
    write_json_f32_array(&mut w, "colors", &colors, 6)?;
    write!(w, "\n")?;
    write!(w, "    }},\n")?;

    // Index data
    write!(w, "    \"index_data\": [")?;
    for (i, idx) in indices.iter().enumerate() {
        if i > 0 {
            write!(w, ", ")?;
        }
        write!(w, "{}", idx)?;
    }
    write!(w, "]\n")?;

    write!(w, "  }}\n}}\n")?;
    w.flush()?;

    Ok(())
}

/// Write a named f32 array as JSON with a given precision
fn write_json_f32_array(
    w: &mut impl Write,
    name: &str,
    data: &[f32],
    precision: usize,
) -> Result<(), IoError> {
    write!(w, "      \"{}\": [", name)?;
    for (i, val) in data.iter().enumerate() {
        if i > 0 {
            write!(w, ", ")?;
        }
        write!(w, "{:.*}", precision, val)?;
    }
    write!(w, "]")?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Binary export
// ---------------------------------------------------------------------------

/// Export mesh as Unity-compatible binary asset (.unity_mesh_bin)
///
/// Binary format:
/// ```text
/// Header (16 bytes):
///   magic: "ALMB" (4 bytes)
///   version: u16 (little-endian)
///   vertex_count: u32 (little-endian)
///   index_count: u32 (little-endian)
///   flags: u16 (little-endian)
///
/// Vertex data (interleaved, per vertex):
///   position: [f32; 3]
///   normal:   [f32; 3]
///   uv:       [f32; 2]
///   tangent:  [f32; 4]
///   color:    [f32; 4]
///   (total: 16 floats = 64 bytes per vertex)
///
/// Index data:
///   indices: [u32; index_count]
/// ```
///
/// # Arguments
/// * `mesh` - Source mesh
/// * `path` - Destination file path
/// * `config` - Unity export configuration
///
/// # Errors
/// Returns `IoError::Io` on filesystem errors.
pub fn export_unity_mesh_binary(
    mesh: &Mesh,
    path: impl AsRef<Path>,
    config: &UnityMeshConfig,
) -> Result<(), IoError> {
    let file = File::create(path)?;
    let mut w = BufWriter::new(file);

    let vc = mesh.vertices.len() as u32;
    let ic = mesh.indices.len() as u32;

    // Flags: match ABM flag definitions for consistency
    let flags: u16 = 0x3F; // all attributes present (normals, uvs, tangents, colors)

    // Header (16 bytes)
    w.write_all(UNITY_BIN_MAGIC)?;
    w.write_all(&UNITY_BIN_VERSION.to_le_bytes())?;
    w.write_all(&vc.to_le_bytes())?;
    w.write_all(&ic.to_le_bytes())?;
    w.write_all(&flags.to_le_bytes())?;

    // Interleaved vertex data
    for v in &mesh.vertices {
        let (px, py, pz) = convert_position(
            v.position.x,
            v.position.y,
            v.position.z,
            config,
        );
        w.write_all(&px.to_le_bytes())?;
        w.write_all(&py.to_le_bytes())?;
        w.write_all(&pz.to_le_bytes())?;

        let (nx, ny, nz) = convert_normal(v.normal.x, v.normal.y, v.normal.z, config);
        w.write_all(&nx.to_le_bytes())?;
        w.write_all(&ny.to_le_bytes())?;
        w.write_all(&nz.to_le_bytes())?;

        w.write_all(&v.uv.x.to_le_bytes())?;
        w.write_all(&v.uv.y.to_le_bytes())?;

        let (tx, ty, tz, tw) =
            convert_tangent(v.tangent.x, v.tangent.y, v.tangent.z, v.tangent.w, config);
        w.write_all(&tx.to_le_bytes())?;
        w.write_all(&ty.to_le_bytes())?;
        w.write_all(&tz.to_le_bytes())?;
        w.write_all(&tw.to_le_bytes())?;

        w.write_all(&v.color[0].to_le_bytes())?;
        w.write_all(&v.color[1].to_le_bytes())?;
        w.write_all(&v.color[2].to_le_bytes())?;
        w.write_all(&v.color[3].to_le_bytes())?;
    }

    // Index data (with winding flip)
    let tri_count = mesh.indices.len() / 3;
    for t in 0..tri_count {
        let base = t * 3;
        let i0 = mesh.indices[base];
        let i1 = mesh.indices[base + 1];
        let i2 = mesh.indices[base + 2];
        if config.flip_winding {
            w.write_all(&i0.to_le_bytes())?;
            w.write_all(&i2.to_le_bytes())?;
            w.write_all(&i1.to_le_bytes())?;
        } else {
            w.write_all(&i0.to_le_bytes())?;
            w.write_all(&i1.to_le_bytes())?;
            w.write_all(&i2.to_le_bytes())?;
        }
    }

    w.flush()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::{sdf_to_mesh, MarchingCubesConfig, Vertex};
    use crate::types::SdfNode;
    use glam::{Vec2, Vec3, Vec4};
    use std::fs;

    fn temp_path(name: &str) -> std::path::PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!("alice_sdf_unity_{}", name));
        path
    }

    /// Build a simple triangle mesh for testing
    fn make_triangle_mesh() -> Mesh {
        let v0 = Vertex::with_all(
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec2::new(0.0, 0.0),
            Vec2::ZERO,
            Vec4::new(1.0, 0.0, 0.0, 1.0),
            [1.0, 0.0, 0.0, 1.0],
            0,
        );
        let v1 = Vertex::with_all(
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::ZERO,
            Vec4::new(0.0, 1.0, 0.0, 1.0),
            [0.0, 1.0, 0.0, 1.0],
            0,
        );
        let v2 = Vertex::with_all(
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec2::new(0.0, 1.0),
            Vec2::ZERO,
            Vec4::new(0.0, 0.0, 1.0, 1.0),
            [0.0, 0.0, 1.0, 1.0],
            0,
        );
        Mesh {
            vertices: vec![v0, v1, v2],
            indices: vec![0, 1, 2],
        }
    }

    #[test]
    fn test_unity_json_export_structure() {
        let mesh = make_triangle_mesh();
        let path = temp_path("structure.unity_mesh");

        export_unity_mesh(&mesh, &path, &UnityMeshConfig::default()).unwrap();

        let content = fs::read_to_string(&path).unwrap();
        assert!(content.contains("\"alice_sdf_mesh\""));
        assert!(content.contains("\"version\": 1"));
        assert!(content.contains("\"vertex_count\": 3"));
        assert!(content.contains("\"index_count\": 3"));
        assert!(content.contains("\"positions\""));
        assert!(content.contains("\"normals\""));
        assert!(content.contains("\"uvs\""));
        assert!(content.contains("\"tangents\""));
        assert!(content.contains("\"colors\""));
        assert!(content.contains("\"index_data\""));
        assert!(content.contains("\"sub_meshes\""));
        assert!(content.contains("\"name\": \"AliceSdfMesh\""));

        fs::remove_file(&path).ok();
    }

    #[test]
    fn test_unity_coordinate_flip() {
        let mesh = make_triangle_mesh();
        let path = temp_path("coord_flip.unity_mesh");

        let config = UnityMeshConfig {
            flip_z: true,
            flip_winding: false, // Disable winding flip to isolate Z flip test
            ..Default::default()
        };
        export_unity_mesh(&mesh, &path, &config).unwrap();

        let content = fs::read_to_string(&path).unwrap();

        // v0 has position z=1.0, after flip should be -1.0
        // Positions array: [0, 0, -1, 1, 0, 0, 0, 1, 0]
        assert!(content.contains("-1.0"));

        // v0 has normal z=1.0, after flip should be -1.0
        // (normals also contain -1.0)

        fs::remove_file(&path).ok();
    }

    #[test]
    fn test_unity_no_flip() {
        let mesh = make_triangle_mesh();
        let path = temp_path("no_flip.unity_mesh");

        let config = UnityMeshConfig {
            flip_z: false,
            flip_winding: false,
            ..Default::default()
        };
        export_unity_mesh(&mesh, &path, &config).unwrap();

        let content = fs::read_to_string(&path).unwrap();

        // v0 has position z=1.0, should remain positive
        assert!(content.contains("\"positions\": [0."));

        fs::remove_file(&path).ok();
    }

    #[test]
    fn test_unity_winding_flip() {
        let mesh = make_triangle_mesh();
        let path = temp_path("winding.unity_mesh");

        let config = UnityMeshConfig {
            flip_z: false,
            flip_winding: true,
            ..Default::default()
        };
        export_unity_mesh(&mesh, &path, &config).unwrap();

        let content = fs::read_to_string(&path).unwrap();

        // Original indices: 0, 1, 2 -> with flip_winding: 0, 2, 1
        assert!(content.contains("\"index_data\": [0, 2, 1]"));

        fs::remove_file(&path).ok();
    }

    #[test]
    fn test_unity_scale_factor() {
        let mesh = make_triangle_mesh();
        let path = temp_path("scale.unity_mesh");

        let config = UnityMeshConfig {
            scale: 2.0,
            flip_z: false,
            flip_winding: false,
            ..Default::default()
        };
        export_unity_mesh(&mesh, &path, &config).unwrap();

        let content = fs::read_to_string(&path).unwrap();

        // v0 position (0,0,1) * 2.0 = (0,0,2)
        // v1 position (1,0,0) * 2.0 = (2,0,0)
        assert!(content.contains("2.0"));

        fs::remove_file(&path).ok();
    }

    #[test]
    fn test_unity_binary_header() {
        let mesh = make_triangle_mesh();
        let path = temp_path("header.unity_mesh_bin");

        export_unity_mesh_binary(&mesh, &path, &UnityMeshConfig::default()).unwrap();

        let data = fs::read(&path).unwrap();

        // Check magic
        assert_eq!(&data[0..4], b"ALMB");
        // Check version
        assert_eq!(u16::from_le_bytes([data[4], data[5]]), 1);
        // Check vertex count
        assert_eq!(u32::from_le_bytes([data[6], data[7], data[8], data[9]]), 3);
        // Check index count
        assert_eq!(
            u32::from_le_bytes([data[10], data[11], data[12], data[13]]),
            3
        );

        // Total file size: 16 (header) + 3 * 64 (vertices) + 3 * 4 (indices)
        let expected = 16 + 3 * 64 + 3 * 4;
        assert_eq!(data.len(), expected);

        fs::remove_file(&path).ok();
    }

    #[test]
    fn test_unity_binary_vertex_data_roundtrip() {
        let mesh = make_triangle_mesh();
        let path = temp_path("roundtrip.unity_mesh_bin");

        let config = UnityMeshConfig {
            flip_z: false,
            flip_winding: false,
            scale: 1.0,
            ..Default::default()
        };
        export_unity_mesh_binary(&mesh, &path, &config).unwrap();

        let data = fs::read(&path).unwrap();

        // Read first vertex position (offset 16)
        let px = f32::from_le_bytes(data[16..20].try_into().unwrap());
        let py = f32::from_le_bytes(data[20..24].try_into().unwrap());
        let pz = f32::from_le_bytes(data[24..28].try_into().unwrap());

        assert!((px - 0.0).abs() < 1e-6);
        assert!((py - 0.0).abs() < 1e-6);
        assert!((pz - 1.0).abs() < 1e-6);

        // Read first vertex normal (offset 16 + 12)
        let nx = f32::from_le_bytes(data[28..32].try_into().unwrap());
        let ny = f32::from_le_bytes(data[32..36].try_into().unwrap());
        let nz = f32::from_le_bytes(data[36..40].try_into().unwrap());

        assert!((nx - 0.0).abs() < 1e-6);
        assert!((ny - 0.0).abs() < 1e-6);
        assert!((nz - 1.0).abs() < 1e-6);

        fs::remove_file(&path).ok();
    }

    #[test]
    fn test_unity_binary_z_flip() {
        let mesh = make_triangle_mesh();
        let path = temp_path("z_flip.unity_mesh_bin");

        let config = UnityMeshConfig {
            flip_z: true,
            flip_winding: false,
            scale: 1.0,
            ..Default::default()
        };
        export_unity_mesh_binary(&mesh, &path, &config).unwrap();

        let data = fs::read(&path).unwrap();

        // First vertex position.z should be flipped: 1.0 -> -1.0
        let pz = f32::from_le_bytes(data[24..28].try_into().unwrap());
        assert!((pz - (-1.0)).abs() < 1e-6);

        // First vertex normal.z should be flipped: 1.0 -> -1.0
        let nz = f32::from_le_bytes(data[36..40].try_into().unwrap());
        assert!((nz - (-1.0)).abs() < 1e-6);

        fs::remove_file(&path).ok();
    }

    #[test]
    fn test_unity_empty_mesh() {
        let mesh = Mesh::new();
        let path_json = temp_path("empty.unity_mesh");
        let path_bin = temp_path("empty.unity_mesh_bin");

        export_unity_mesh(&mesh, &path_json, &UnityMeshConfig::default()).unwrap();
        export_unity_mesh_binary(&mesh, &path_bin, &UnityMeshConfig::default()).unwrap();

        let content = fs::read_to_string(&path_json).unwrap();
        assert!(content.contains("\"vertex_count\": 0"));
        assert!(content.contains("\"index_count\": 0"));

        let data = fs::read(&path_bin).unwrap();
        // Header only (16 bytes) + 0 vertices + 0 indices
        assert_eq!(data.len(), 16);

        fs::remove_file(&path_json).ok();
        fs::remove_file(&path_bin).ok();
    }

    #[test]
    fn test_unity_sphere_export() {
        let sphere = SdfNode::sphere(1.0);
        let mc_config = MarchingCubesConfig {
            resolution: 8,
            iso_level: 0.0,
            compute_normals: true,
            ..Default::default()
        };
        let mesh = sdf_to_mesh(&sphere, Vec3::splat(-2.0), Vec3::splat(2.0), &mc_config);

        let path = temp_path("sphere.unity_mesh");
        export_unity_mesh(&mesh, &path, &UnityMeshConfig::default()).unwrap();

        let content = fs::read_to_string(&path).unwrap();
        assert!(content.contains("\"alice_sdf_mesh\""));
        assert!(content.contains("\"vertex_count\""));

        // Verify vertex count is positive
        let vc_start = content.find("\"vertex_count\": ").unwrap() + "\"vertex_count\": ".len();
        let vc_end = content[vc_start..].find(',').unwrap() + vc_start;
        let vc: usize = content[vc_start..vc_end].parse().unwrap();
        assert!(vc > 0, "Sphere mesh should have vertices");

        fs::remove_file(&path).ok();
    }

    #[test]
    fn test_unity_custom_name() {
        let mesh = make_triangle_mesh();
        let path = temp_path("named.unity_mesh");

        let config = UnityMeshConfig {
            name: "MyCustomMesh".to_string(),
            ..Default::default()
        };
        export_unity_mesh(&mesh, &path, &config).unwrap();

        let content = fs::read_to_string(&path).unwrap();
        assert!(content.contains("\"name\": \"MyCustomMesh\""));

        fs::remove_file(&path).ok();
    }
}

//! UE5 mesh asset export (Deep Fried Edition)
//!
//! Generates UE5-compatible mesh data for import into Unreal Engine 5.
//! Supports single-mesh export, binary export, and multi-LOD StaticMesh export.
//!
//! # Formats
//!
//! - `.ue5_mesh` (JSON): Human-readable asset with LOD support
//! - `.ue5_mesh_bin` (Binary): Compact vertex/index data for runtime loading
//!
//! # Coordinate Conversion
//!
//! UE5 uses a **left-handed** coordinate system with **Z-up** and centimeters.
//! ALICE-SDF uses a right-handed Y-up coordinate system in meters.
//! - Swap Y and Z: `(x, y, z) -> (x, z, y)`
//! - Scale by 100: meters to centimeters
//! - Flip winding order for left-handed
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

/// Magic bytes for UE5 binary mesh format
const UE5_BIN_MAGIC: &[u8; 4] = b"ALUE";

/// UE5 binary mesh version
const UE5_BIN_VERSION: u16 = 1;

/// JSON format version
const UE5_JSON_VERSION: u32 = 1;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// UE5 mesh export configuration
#[derive(Debug, Clone)]
pub struct Ue5MeshConfig {
    /// Scale factor (UE5 uses centimeters; default 100.0 converts meters to cm)
    pub scale: f32,
    /// Asset name
    pub name: String,
    /// LOD index (0 = highest detail)
    pub lod_index: u32,
}

impl Default for Ue5MeshConfig {
    fn default() -> Self {
        Self {
            scale: 100.0,
            name: "SM_AliceSdf".to_string(),
            lod_index: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Coordinate helpers
// ---------------------------------------------------------------------------

/// Convert position from ALICE-SDF (right-handed Y-up, meters) to UE5
/// (left-handed Z-up, centimeters).
///
/// Mapping: (x, y, z) -> (x * scale, z * scale, y * scale)
#[inline(always)]
fn convert_position(x: f32, y: f32, z: f32, scale: f32) -> (f32, f32, f32) {
    (x * scale, z * scale, y * scale)
}

/// Convert normal/tangent direction from ALICE-SDF to UE5.
///
/// Mapping: (x, y, z) -> (x, z, y)  (no scale for unit vectors)
#[inline(always)]
fn convert_direction(x: f32, y: f32, z: f32) -> (f32, f32, f32) {
    (x, z, y)
}

/// Convert tangent vector (xyz direction + w handedness).
#[inline(always)]
fn convert_tangent(x: f32, y: f32, z: f32, w: f32) -> (f32, f32, f32, f32) {
    // Flip handedness for left-handed coordinate system
    (x, z, y, -w)
}

/// Compute axis-aligned bounding box for a mesh (in UE5 coordinates).
/// Returns (origin, extent) where origin is the center and extent is the half-size.
fn compute_bounds(mesh: &Mesh, scale: f32) -> ([f32; 3], [f32; 3]) {
    if mesh.vertices.is_empty() {
        return ([0.0; 3], [0.0; 3]);
    }

    let mut min = [f32::MAX; 3];
    let mut max = [f32::MIN; 3];

    for v in &mesh.vertices {
        let (ux, uy, uz) = convert_position(v.position.x, v.position.y, v.position.z, scale);
        min[0] = min[0].min(ux);
        min[1] = min[1].min(uy);
        min[2] = min[2].min(uz);
        max[0] = max[0].max(ux);
        max[1] = max[1].max(uy);
        max[2] = max[2].max(uz);
    }

    let origin = [
        (min[0] + max[0]) * 0.5,
        (min[1] + max[1]) * 0.5,
        (min[2] + max[2]) * 0.5,
    ];
    let extent = [
        (max[0] - min[0]) * 0.5,
        (max[1] - min[1]) * 0.5,
        (max[2] - min[2]) * 0.5,
    ];
    (origin, extent)
}

// ---------------------------------------------------------------------------
// JSON export (single LOD)
// ---------------------------------------------------------------------------

/// Export mesh as UE5-compatible JSON asset (.ue5_mesh)
///
/// Generates a JSON manifest matching UE5 StaticMesh structure with a single
/// LOD level. Coordinates are converted to UE5's left-handed Z-up system
/// with centimeter scale.
///
/// # Arguments
/// * `mesh` - Source mesh
/// * `path` - Destination file path
/// * `config` - UE5 export configuration
///
/// # Errors
/// Returns `IoError::Io` on filesystem errors.
pub fn export_ue5_mesh(
    mesh: &Mesh,
    path: impl AsRef<Path>,
    config: &Ue5MeshConfig,
) -> Result<(), IoError> {
    export_ue5_mesh_with_lods(&[mesh.clone()], &[1.0], path, config)
}

// ---------------------------------------------------------------------------
// Binary export
// ---------------------------------------------------------------------------

/// Export mesh as UE5-compatible binary asset (.ue5_mesh_bin)
///
/// Binary format:
/// ```text
/// Header (16 bytes):
///   magic: "ALUE" (4 bytes)
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
/// * `config` - UE5 export configuration
///
/// # Errors
/// Returns `IoError::Io` on filesystem errors.
pub fn export_ue5_mesh_binary(
    mesh: &Mesh,
    path: impl AsRef<Path>,
    config: &Ue5MeshConfig,
) -> Result<(), IoError> {
    let file = File::create(path)?;
    let mut w = BufWriter::new(file);

    let vc = mesh.vertices.len() as u32;
    let ic = mesh.indices.len() as u32;
    let flags: u16 = 0x3F;

    // Header (16 bytes)
    w.write_all(UE5_BIN_MAGIC)?;
    w.write_all(&UE5_BIN_VERSION.to_le_bytes())?;
    w.write_all(&vc.to_le_bytes())?;
    w.write_all(&ic.to_le_bytes())?;
    w.write_all(&flags.to_le_bytes())?;

    // Interleaved vertex data
    for v in &mesh.vertices {
        let (px, py, pz) =
            convert_position(v.position.x, v.position.y, v.position.z, config.scale);
        w.write_all(&px.to_le_bytes())?;
        w.write_all(&py.to_le_bytes())?;
        w.write_all(&pz.to_le_bytes())?;

        let (nx, ny, nz) = convert_direction(v.normal.x, v.normal.y, v.normal.z);
        w.write_all(&nx.to_le_bytes())?;
        w.write_all(&ny.to_le_bytes())?;
        w.write_all(&nz.to_le_bytes())?;

        w.write_all(&v.uv.x.to_le_bytes())?;
        w.write_all(&v.uv.y.to_le_bytes())?;

        let (tx, ty, tz, tw) =
            convert_tangent(v.tangent.x, v.tangent.y, v.tangent.z, v.tangent.w);
        w.write_all(&tx.to_le_bytes())?;
        w.write_all(&ty.to_le_bytes())?;
        w.write_all(&tz.to_le_bytes())?;
        w.write_all(&tw.to_le_bytes())?;

        w.write_all(&v.color[0].to_le_bytes())?;
        w.write_all(&v.color[1].to_le_bytes())?;
        w.write_all(&v.color[2].to_le_bytes())?;
        w.write_all(&v.color[3].to_le_bytes())?;
    }

    // Index data (flip winding for left-handed)
    let tri_count = mesh.indices.len() / 3;
    for t in 0..tri_count {
        let base = t * 3;
        let i0 = mesh.indices[base];
        let i1 = mesh.indices[base + 1];
        let i2 = mesh.indices[base + 2];
        // Flip winding for left-handed
        w.write_all(&i0.to_le_bytes())?;
        w.write_all(&i2.to_le_bytes())?;
        w.write_all(&i1.to_le_bytes())?;
    }

    w.flush()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Multi-LOD JSON export
// ---------------------------------------------------------------------------

/// Export multiple LODs as UE5 StaticMesh-style JSON asset
///
/// Each mesh in `meshes` represents one LOD level (index 0 = highest detail).
/// `lod_screen_sizes` specifies the screen-size threshold for each LOD level.
///
/// # Arguments
/// * `meshes` - LOD meshes from highest to lowest detail
/// * `lod_screen_sizes` - Screen-size threshold for each LOD level
/// * `path` - Destination file path
/// * `config` - UE5 export configuration
///
/// # Errors
/// Returns `IoError::InvalidFormat` if `meshes` is empty or `lod_screen_sizes`
/// length does not match, `IoError::Io` on filesystem errors.
pub fn export_ue5_mesh_with_lods(
    meshes: &[Mesh],
    lod_screen_sizes: &[f32],
    path: impl AsRef<Path>,
    config: &Ue5MeshConfig,
) -> Result<(), IoError> {
    if meshes.is_empty() {
        return Err(IoError::InvalidFormat("No meshes provided".into()));
    }
    if lod_screen_sizes.len() != meshes.len() {
        return Err(IoError::InvalidFormat(format!(
            "Expected {} screen sizes for {} LODs, got {}",
            meshes.len(),
            meshes.len(),
            lod_screen_sizes.len()
        )));
    }

    let file = File::create(path)?;
    let mut w = BufWriter::new(file);

    // Compute global bounds from LOD 0
    let (origin, extent) = compute_bounds(&meshes[0], config.scale);

    write!(w, "{{\n  \"alice_sdf_static_mesh\": {{\n")?;
    write!(w, "    \"version\": {},\n", UE5_JSON_VERSION)?;
    write!(w, "    \"name\": \"{}\",\n", config.name)?;

    // LODs array
    write!(w, "    \"lods\": [\n")?;

    for (lod_idx, mesh) in meshes.iter().enumerate() {
        if lod_idx > 0 {
            write!(w, ",\n")?;
        }

        let ic = mesh.indices.len();
        let tri_count = ic / 3;

        // Compute min/max vertex indices
        let min_vi = if mesh.indices.is_empty() {
            0
        } else {
            *mesh.indices.iter().min().unwrap()
        };
        let max_vi = if mesh.indices.is_empty() {
            0
        } else {
            *mesh.indices.iter().max().unwrap()
        };

        write!(w, "      {{\n")?;
        write!(w, "        \"lod_index\": {},\n", lod_idx)?;
        write!(
            w,
            "        \"screen_size\": {:.6},\n",
            lod_screen_sizes[lod_idx]
        )?;

        // Sections (single section per LOD)
        write!(w, "        \"sections\": [{{\n")?;
        write!(w, "          \"material_index\": 0,\n")?;
        write!(w, "          \"first_index\": 0,\n")?;
        write!(w, "          \"num_triangles\": {},\n", tri_count)?;
        write!(w, "          \"min_vertex_index\": {},\n", min_vi)?;
        write!(w, "          \"max_vertex_index\": {}\n", max_vi)?;
        write!(w, "        }}],\n")?;

        // Vertex data
        write!(w, "        \"vertex_data\": {{\n")?;

        // Positions
        write_ue5_vertex_positions(&mut w, mesh, config.scale)?;
        write!(w, ",\n")?;

        // Normals
        write_ue5_vertex_normals(&mut w, mesh)?;
        write!(w, ",\n")?;

        // UVs (two channels: primary + lightmap)
        write_ue5_vertex_uvs(&mut w, mesh)?;
        write!(w, ",\n")?;

        // Tangents
        write_ue5_vertex_tangents(&mut w, mesh)?;
        write!(w, "\n")?;

        write!(w, "        }},\n")?;

        // Index data (flip winding for left-handed)
        write!(w, "        \"index_data\": [")?;
        for t in 0..tri_count {
            if t > 0 {
                write!(w, ", ")?;
            }
            let base = t * 3;
            let i0 = mesh.indices[base];
            let i1 = mesh.indices[base + 1];
            let i2 = mesh.indices[base + 2];
            // Flip winding
            write!(w, "{}, {}, {}", i0, i2, i1)?;
        }
        write!(w, "]\n")?;

        write!(w, "      }}")?;
    }

    write!(w, "\n    ],\n")?;

    // Bounds
    write!(w, "    \"bounds\": {{\n")?;
    write!(
        w,
        "      \"origin\": [{:.6}, {:.6}, {:.6}],\n",
        origin[0], origin[1], origin[2]
    )?;
    write!(
        w,
        "      \"extent\": [{:.6}, {:.6}, {:.6}]\n",
        extent[0], extent[1], extent[2]
    )?;
    write!(w, "    }}\n")?;

    write!(w, "  }}\n}}\n")?;
    w.flush()?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Vertex data writers
// ---------------------------------------------------------------------------

fn write_ue5_vertex_positions(
    w: &mut impl Write,
    mesh: &Mesh,
    scale: f32,
) -> Result<(), IoError> {
    write!(w, "          \"positions\": [")?;
    for (i, v) in mesh.vertices.iter().enumerate() {
        if i > 0 {
            write!(w, ", ")?;
        }
        let (ux, uy, uz) = convert_position(v.position.x, v.position.y, v.position.z, scale);
        write!(w, "{:.6}, {:.6}, {:.6}", ux, uy, uz)?;
    }
    write!(w, "]")?;
    Ok(())
}

fn write_ue5_vertex_normals(w: &mut impl Write, mesh: &Mesh) -> Result<(), IoError> {
    write!(w, "          \"normals\": [")?;
    for (i, v) in mesh.vertices.iter().enumerate() {
        if i > 0 {
            write!(w, ", ")?;
        }
        let (nx, ny, nz) = convert_direction(v.normal.x, v.normal.y, v.normal.z);
        write!(w, "{:.6}, {:.6}, {:.6}", nx, ny, nz)?;
    }
    write!(w, "]")?;
    Ok(())
}

fn write_ue5_vertex_uvs(w: &mut impl Write, mesh: &Mesh) -> Result<(), IoError> {
    // UE5 supports multiple UV channels as nested arrays
    write!(w, "          \"uvs\": [[")?;
    // Channel 0: primary UV
    for (i, v) in mesh.vertices.iter().enumerate() {
        if i > 0 {
            write!(w, ", ")?;
        }
        write!(w, "{:.6}, {:.6}", v.uv.x, v.uv.y)?;
    }
    write!(w, "], [")?;
    // Channel 1: lightmap UV (uv2)
    for (i, v) in mesh.vertices.iter().enumerate() {
        if i > 0 {
            write!(w, ", ")?;
        }
        write!(w, "{:.6}, {:.6}", v.uv2.x, v.uv2.y)?;
    }
    write!(w, "]]")?;
    Ok(())
}

fn write_ue5_vertex_tangents(w: &mut impl Write, mesh: &Mesh) -> Result<(), IoError> {
    write!(w, "          \"tangents\": [")?;
    for (i, v) in mesh.vertices.iter().enumerate() {
        if i > 0 {
            write!(w, ", ")?;
        }
        let (tx, ty, tz, tw) =
            convert_tangent(v.tangent.x, v.tangent.y, v.tangent.z, v.tangent.w);
        write!(w, "{:.6}, {:.6}, {:.6}, {:.6}", tx, ty, tz, tw)?;
    }
    write!(w, "]")?;
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
        path.push(format!("alice_sdf_ue5_{}", name));
        path
    }

    /// Build a simple triangle mesh for testing
    fn make_triangle_mesh() -> Mesh {
        let v0 = Vertex::with_all(
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec2::new(0.0, 0.0),
            Vec2::new(0.1, 0.2),
            Vec4::new(1.0, 0.0, 0.0, 1.0),
            [1.0, 0.0, 0.0, 1.0],
            0,
        );
        let v1 = Vertex::with_all(
            Vec3::new(4.0, 5.0, 6.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(0.3, 0.4),
            Vec4::new(0.0, 1.0, 0.0, 1.0),
            [0.0, 1.0, 0.0, 1.0],
            0,
        );
        let v2 = Vertex::with_all(
            Vec3::new(7.0, 8.0, 9.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec2::new(0.0, 1.0),
            Vec2::new(0.5, 0.6),
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
    fn test_ue5_json_export_structure() {
        let mesh = make_triangle_mesh();
        let path = temp_path("structure.ue5_mesh");

        export_ue5_mesh(&mesh, &path, &Ue5MeshConfig::default()).unwrap();

        let content = fs::read_to_string(&path).unwrap();
        assert!(content.contains("\"alice_sdf_static_mesh\""));
        assert!(content.contains("\"version\": 1"));
        assert!(content.contains("\"name\": \"SM_AliceSdf\""));
        assert!(content.contains("\"lods\""));
        assert!(content.contains("\"lod_index\": 0"));
        assert!(content.contains("\"screen_size\""));
        assert!(content.contains("\"sections\""));
        assert!(content.contains("\"positions\""));
        assert!(content.contains("\"normals\""));
        assert!(content.contains("\"uvs\""));
        assert!(content.contains("\"tangents\""));
        assert!(content.contains("\"index_data\""));
        assert!(content.contains("\"bounds\""));
        assert!(content.contains("\"origin\""));
        assert!(content.contains("\"extent\""));

        fs::remove_file(&path).ok();
    }

    #[test]
    fn test_ue5_scale_factor() {
        let mesh = make_triangle_mesh();
        let path = temp_path("scale.ue5_mesh");

        // Default scale is 100 (meters -> cm)
        export_ue5_mesh(&mesh, &path, &Ue5MeshConfig::default()).unwrap();

        let content = fs::read_to_string(&path).unwrap();

        // v0 position (1, 2, 3) -> UE5: (1*100, 3*100, 2*100) = (100, 300, 200)
        assert!(content.contains("100.000000"));
        assert!(content.contains("300.000000"));
        assert!(content.contains("200.000000"));

        fs::remove_file(&path).ok();
    }

    #[test]
    fn test_ue5_coordinate_swap() {
        let mesh = make_triangle_mesh();
        let path = temp_path("coord_swap.ue5_mesh");

        let config = Ue5MeshConfig {
            scale: 1.0, // Use 1.0 to isolate coordinate swap
            ..Default::default()
        };
        export_ue5_mesh(&mesh, &path, &config).unwrap();

        let content = fs::read_to_string(&path).unwrap();

        // v0 position (1, 2, 3) -> UE5: (1, 3, 2) (Y and Z swapped)
        // positions array should contain: 1.0, 3.0, 2.0 for v0
        assert!(content.contains("1.000000, 3.000000, 2.000000"));

        // v0 normal (0, 1, 0) -> UE5: (0, 0, 1) (Y and Z swapped)
        assert!(content.contains("0.000000, 0.000000, 1.000000"));

        fs::remove_file(&path).ok();
    }

    #[test]
    fn test_ue5_winding_flip() {
        let mesh = make_triangle_mesh();
        let path = temp_path("winding.ue5_mesh");

        let config = Ue5MeshConfig {
            scale: 1.0,
            ..Default::default()
        };
        export_ue5_mesh(&mesh, &path, &config).unwrap();

        let content = fs::read_to_string(&path).unwrap();

        // Original indices: 0, 1, 2 -> flipped winding: 0, 2, 1
        assert!(content.contains("\"index_data\": [0, 2, 1]"));

        fs::remove_file(&path).ok();
    }

    #[test]
    fn test_ue5_lod_export() {
        // Create two LODs
        let lod0 = make_triangle_mesh();
        let lod1 = Mesh {
            vertices: vec![Vertex::new(Vec3::ZERO, Vec3::Y)],
            indices: vec![0, 0, 0],
        };

        let path = temp_path("lods.ue5_mesh");
        let screen_sizes = vec![1.0, 0.5];
        let config = Ue5MeshConfig {
            scale: 1.0,
            ..Default::default()
        };

        export_ue5_mesh_with_lods(&[lod0, lod1], &screen_sizes, &path, &config).unwrap();

        let content = fs::read_to_string(&path).unwrap();
        assert!(content.contains("\"lod_index\": 0"));
        assert!(content.contains("\"lod_index\": 1"));
        assert!(content.contains("\"screen_size\": 1.000000"));
        assert!(content.contains("\"screen_size\": 0.500000"));

        fs::remove_file(&path).ok();
    }

    #[test]
    fn test_ue5_binary_header() {
        let mesh = make_triangle_mesh();
        let path = temp_path("header.ue5_mesh_bin");

        export_ue5_mesh_binary(&mesh, &path, &Ue5MeshConfig::default()).unwrap();

        let data = fs::read(&path).unwrap();

        // Check magic
        assert_eq!(&data[0..4], b"ALUE");
        // Check version
        assert_eq!(u16::from_le_bytes([data[4], data[5]]), 1);
        // Check vertex count
        assert_eq!(u32::from_le_bytes([data[6], data[7], data[8], data[9]]), 3);
        // Check index count
        assert_eq!(
            u32::from_le_bytes([data[10], data[11], data[12], data[13]]),
            3
        );

        // Total size: 16 (header) + 3*64 (vertices) + 3*4 (indices)
        let expected = 16 + 3 * 64 + 3 * 4;
        assert_eq!(data.len(), expected);

        fs::remove_file(&path).ok();
    }

    #[test]
    fn test_ue5_binary_position_conversion() {
        let mesh = make_triangle_mesh();
        let path = temp_path("pos_conv.ue5_mesh_bin");

        let config = Ue5MeshConfig {
            scale: 100.0,
            ..Default::default()
        };
        export_ue5_mesh_binary(&mesh, &path, &config).unwrap();

        let data = fs::read(&path).unwrap();

        // First vertex position: (1,2,3) -> UE5: (100, 300, 200)
        let px = f32::from_le_bytes(data[16..20].try_into().unwrap());
        let py = f32::from_le_bytes(data[20..24].try_into().unwrap());
        let pz = f32::from_le_bytes(data[24..28].try_into().unwrap());

        assert!((px - 100.0).abs() < 1e-3);
        assert!((py - 300.0).abs() < 1e-3); // z*100
        assert!((pz - 200.0).abs() < 1e-3); // y*100

        fs::remove_file(&path).ok();
    }

    #[test]
    fn test_ue5_binary_normal_conversion() {
        let mesh = make_triangle_mesh();
        let path = temp_path("norm_conv.ue5_mesh_bin");

        let config = Ue5MeshConfig {
            scale: 1.0,
            ..Default::default()
        };
        export_ue5_mesh_binary(&mesh, &path, &config).unwrap();

        let data = fs::read(&path).unwrap();

        // First vertex normal: (0,1,0) -> UE5: (0, 0, 1)
        let nx = f32::from_le_bytes(data[28..32].try_into().unwrap());
        let ny = f32::from_le_bytes(data[32..36].try_into().unwrap());
        let nz = f32::from_le_bytes(data[36..40].try_into().unwrap());

        assert!((nx - 0.0).abs() < 1e-6);
        assert!((ny - 0.0).abs() < 1e-6); // y=1 goes to z position
        assert!((nz - 1.0).abs() < 1e-6); // z=0 goes to y position, y=1 goes to z

        fs::remove_file(&path).ok();
    }

    #[test]
    fn test_ue5_empty_mesh() {
        let mesh = Mesh::new();
        let path_json = temp_path("empty.ue5_mesh");
        let path_bin = temp_path("empty.ue5_mesh_bin");

        export_ue5_mesh(&mesh, &path_json, &Ue5MeshConfig::default()).unwrap();
        export_ue5_mesh_binary(&mesh, &path_bin, &Ue5MeshConfig::default()).unwrap();

        let content = fs::read_to_string(&path_json).unwrap();
        assert!(content.contains("\"alice_sdf_static_mesh\""));

        let data = fs::read(&path_bin).unwrap();
        assert_eq!(data.len(), 16); // Header only

        fs::remove_file(&path_json).ok();
        fs::remove_file(&path_bin).ok();
    }

    #[test]
    fn test_ue5_sphere_export() {
        let sphere = SdfNode::sphere(1.0);
        let mc_config = MarchingCubesConfig {
            resolution: 8,
            iso_level: 0.0,
            compute_normals: true,
            ..Default::default()
        };
        let mesh = sdf_to_mesh(&sphere, Vec3::splat(-2.0), Vec3::splat(2.0), &mc_config);

        let path = temp_path("sphere.ue5_mesh");
        export_ue5_mesh(&mesh, &path, &Ue5MeshConfig::default()).unwrap();

        let content = fs::read_to_string(&path).unwrap();
        assert!(content.contains("\"alice_sdf_static_mesh\""));
        assert!(content.contains("\"bounds\""));

        fs::remove_file(&path).ok();
    }

    #[test]
    fn test_ue5_bounds_calculation() {
        let mesh = make_triangle_mesh();
        let path = temp_path("bounds.ue5_mesh");

        let config = Ue5MeshConfig {
            scale: 1.0,
            ..Default::default()
        };
        export_ue5_mesh(&mesh, &path, &config).unwrap();

        let content = fs::read_to_string(&path).unwrap();

        // Positions in UE5 coords (scale=1):
        // v0: (1, 3, 2)
        // v1: (4, 6, 5)
        // v2: (7, 9, 8)
        // min: (1, 3, 2), max: (7, 9, 8)
        // origin: (4, 6, 5), extent: (3, 3, 3)
        assert!(content.contains("\"origin\": [4.000000, 6.000000, 5.000000]"));
        assert!(content.contains("\"extent\": [3.000000, 3.000000, 3.000000]"));

        fs::remove_file(&path).ok();
    }

    #[test]
    fn test_ue5_custom_name() {
        let mesh = make_triangle_mesh();
        let path = temp_path("named.ue5_mesh");

        let config = Ue5MeshConfig {
            name: "SM_MyCustomMesh".to_string(),
            ..Default::default()
        };
        export_ue5_mesh(&mesh, &path, &config).unwrap();

        let content = fs::read_to_string(&path).unwrap();
        assert!(content.contains("\"name\": \"SM_MyCustomMesh\""));

        fs::remove_file(&path).ok();
    }

    #[test]
    fn test_ue5_lod_screen_sizes_mismatch() {
        let mesh = make_triangle_mesh();
        let path = temp_path("mismatch.ue5_mesh");

        // 1 mesh but 2 screen sizes = error
        let result =
            export_ue5_mesh_with_lods(&[mesh], &[1.0, 0.5], &path, &Ue5MeshConfig::default());
        assert!(matches!(result, Err(IoError::InvalidFormat(_))));
    }

    #[test]
    fn test_ue5_empty_meshes_error() {
        let path = temp_path("no_meshes.ue5_mesh");
        let result =
            export_ue5_mesh_with_lods(&[], &[], &path, &Ue5MeshConfig::default());
        assert!(matches!(result, Err(IoError::InvalidFormat(_))));
    }

    #[test]
    fn test_ue5_section_vertex_range() {
        let mesh = make_triangle_mesh();
        let path = temp_path("section.ue5_mesh");

        let config = Ue5MeshConfig {
            scale: 1.0,
            ..Default::default()
        };
        export_ue5_mesh(&mesh, &path, &config).unwrap();

        let content = fs::read_to_string(&path).unwrap();
        assert!(content.contains("\"min_vertex_index\": 0"));
        assert!(content.contains("\"max_vertex_index\": 2"));
        assert!(content.contains("\"num_triangles\": 1"));

        fs::remove_file(&path).ok();
    }
}

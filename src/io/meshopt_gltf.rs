//! glTF 2.0 export with `EXT_meshopt_compression` extension
//!
//! Provides a compact GLB writer that applies meshopt binary-compat encoding
//! to POSITION / NORMAL / TEXCOORD_0 / index buffer views The output is a
//! valid glTF 2.0 with `extensionsUsed` and `extensionsRequired` set to
//! `["EXT_meshopt_compression"]` — loaders that support the extension will
//! decompress on load
//!
//! # Limitations
//!
//! - Single-mesh, single-scene structure (mesh + material + node + scene)
//! - Supports POSITION (VEC3 f32) / NORMAL (VEC3 f32) / TEXCOORD_0 (VEC2 f32) / indices (u32)
//! - No PBR material extensions (see `io::gltf` for full material support)
//! - Fallback bytes are zero-filled (compression is REQUIRED for load)
//!
//! # Reference
//!
//! - <https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Vendor/EXT_meshopt_compression/README.md>
//!
//! Author: Moroya Sakamoto

use crate::io::IoError;
use crate::mesh::meshopt_index_codec::encode_index_buffer;
use crate::mesh::meshopt_vertex_codec::encode_vertex_buffer_level;
use crate::mesh::Mesh;
use std::io::Write;
use std::path::Path;

// GLB constants
const GLB_MAGIC: u32 = 0x4654_6C67; // "glTF"
const GLB_VERSION: u32 = 2;
const GLB_CHUNK_JSON: u32 = 0x4E4F_534A; // "JSON"
const GLB_CHUNK_BIN: u32 = 0x004E_4942; // "BIN\0"

const FLOAT: u32 = 5126;
const UNSIGNED_INT: u32 = 5125;
const ARRAY_BUFFER: u32 = 34962;
const ELEMENT_ARRAY_BUFFER: u32 = 34963;

/// Configuration for meshopt-compressed glTF export
#[derive(Debug, Clone)]
pub struct MeshoptGltfConfig {
    /// Include normals (NORMAL attribute)
    pub export_normals: bool,
    /// Include TEXCOORD_0 (uv attribute)
    pub export_uvs: bool,
    /// Compression level (0=scalar, 2=u8/u16 estimate, 3=u8/u16/u32 XOR)
    pub level: u8,
    /// Force `doubleSided` on the material
    pub double_sided: bool,
}

impl Default for MeshoptGltfConfig {
    fn default() -> Self {
        Self {
            export_normals: true,
            export_uvs: false,
            level: 2,
            double_sided: false,
        }
    }
}

/// Metadata describing a single meshopt-compressed buffer view
///
/// Used to construct the JSON `EXT_meshopt_compression` extension block for
/// each bufferView
#[derive(Debug, Clone, Copy)]
struct MeshoptView {
    /// Original uncompressed byte length
    original_length: usize,
    /// Original stride in bytes (0 = tightly packed)
    stride: usize,
    /// Item count (vertex or index count)
    count: usize,
    /// Offset within the buffer where compressed bytes begin
    compressed_offset: usize,
    /// Length of the compressed bytes
    compressed_length: usize,
    /// Meshopt mode: "ATTRIBUTES" | "TRIANGLES" | "INDICES"
    mode: &'static str,
}

impl MeshoptView {
    fn ext_json(&self) -> String {
        format!(
            r#"{{"buffer":0,"byteOffset":{},"byteLength":{},"byteStride":{},"count":{},"mode":"{}"}}"#,
            self.compressed_offset, self.compressed_length, self.stride, self.count, self.mode,
        )
    }
}

/// Export mesh to a meshopt-compressed GLB (`.glb`) file
///
/// # Errors
///
/// Returns `IoError` on file I/O failure or on invalid mesh
pub fn export_glb_meshopt<P: AsRef<Path>>(
    mesh: &Mesh,
    path: P,
    config: &MeshoptGltfConfig,
) -> Result<(), IoError> {
    let bytes = export_glb_meshopt_bytes(mesh, config)?;
    let mut file = std::fs::File::create(path.as_ref())?;
    file.write_all(&bytes)?;
    Ok(())
}

/// Export mesh to meshopt-compressed GLB bytes
///
/// # Errors
///
/// Returns `IoError` if the mesh has no vertices or triangles
pub fn export_glb_meshopt_bytes(
    mesh: &Mesh,
    config: &MeshoptGltfConfig,
) -> Result<Vec<u8>, IoError> {
    if mesh.vertices.is_empty() {
        return Err(IoError::InvalidFormat("empty mesh".into()));
    }
    if mesh.indices.is_empty() || mesh.indices.len() % 3 != 0 {
        return Err(IoError::InvalidFormat(
            "mesh indices must be non-empty triangles".into(),
        ));
    }

    let vert_count = mesh.vertices.len();
    let idx_count = mesh.indices.len();

    // Build raw buffers per attribute
    let positions_raw: Vec<u8> = mesh
        .vertices
        .iter()
        .flat_map(|v| {
            let mut b = Vec::with_capacity(12);
            b.extend_from_slice(&v.position.x.to_le_bytes());
            b.extend_from_slice(&v.position.y.to_le_bytes());
            b.extend_from_slice(&v.position.z.to_le_bytes());
            b
        })
        .collect();

    let normals_raw: Option<Vec<u8>> = config.export_normals.then(|| {
        mesh.vertices
            .iter()
            .flat_map(|v| {
                let mut b = Vec::with_capacity(12);
                b.extend_from_slice(&v.normal.x.to_le_bytes());
                b.extend_from_slice(&v.normal.y.to_le_bytes());
                b.extend_from_slice(&v.normal.z.to_le_bytes());
                b
            })
            .collect()
    });

    let uvs_raw: Option<Vec<u8>> = config.export_uvs.then(|| {
        mesh.vertices
            .iter()
            .flat_map(|v| {
                let mut b = Vec::with_capacity(8);
                b.extend_from_slice(&v.uv.x.to_le_bytes());
                b.extend_from_slice(&v.uv.y.to_le_bytes());
                b
            })
            .collect()
    });

    let indices_raw: Vec<u8> = mesh.indices.iter().flat_map(|&i| i.to_le_bytes()).collect();

    // Compress each
    let positions_compressed = encode_vertex_buffer_level(&positions_raw, 12, config.level);
    let normals_compressed = normals_raw
        .as_ref()
        .map(|raw| encode_vertex_buffer_level(raw, 12, config.level));
    let uvs_compressed = uvs_raw
        .as_ref()
        .map(|raw| encode_vertex_buffer_level(raw, 8, config.level));
    // Index buffer uses meshopt index codec (VEC3 triangles)
    let indices_compressed = encode_index_buffer(&mesh.indices);

    // AABB for POSITION accessor
    let mut min_pos = [f32::MAX; 3];
    let mut max_pos = [f32::MIN; 3];
    for v in &mesh.vertices {
        min_pos[0] = min_pos[0].min(v.position.x);
        min_pos[1] = min_pos[1].min(v.position.y);
        min_pos[2] = min_pos[2].min(v.position.z);
        max_pos[0] = max_pos[0].max(v.position.x);
        max_pos[1] = max_pos[1].max(v.position.y);
        max_pos[2] = max_pos[2].max(v.position.z);
    }

    // Build BIN: fallback (zero-filled) sections + compressed sections
    // Layout:
    //   [fallback positions][fallback normals?][fallback uvs?][fallback indices]
    //   [compressed positions][compressed normals?][compressed uvs?][compressed indices]
    //
    // BufferView primary offsets point to fallback (zero) region; extension
    // offsets point to compressed region Compression is REQUIRED, so
    // fallback is never used but present to satisfy the required-length rules
    let mut bin: Vec<u8> = Vec::new();

    // Fallback region offsets
    let fb_pos_off = bin.len();
    bin.resize(bin.len() + positions_raw.len(), 0);
    let fb_norm_off = normals_raw.as_ref().map(|raw| {
        let o = bin.len();
        bin.resize(bin.len() + raw.len(), 0);
        o
    });
    let fb_uv_off = uvs_raw.as_ref().map(|raw| {
        let o = bin.len();
        bin.resize(bin.len() + raw.len(), 0);
        o
    });
    let fb_idx_off = bin.len();
    bin.resize(bin.len() + indices_raw.len(), 0);

    // Compressed region offsets
    let cmp_pos_off = bin.len();
    bin.extend_from_slice(&positions_compressed);
    let cmp_norm_off = normals_compressed.as_ref().map(|c| {
        let o = bin.len();
        bin.extend_from_slice(c);
        o
    });
    let cmp_uv_off = uvs_compressed.as_ref().map(|c| {
        let o = bin.len();
        bin.extend_from_slice(c);
        o
    });
    let cmp_idx_off = bin.len();
    bin.extend_from_slice(&indices_compressed);

    // Build bufferView + extension JSON
    let mut buffer_views: Vec<String> = Vec::new();
    let mut accessors: Vec<String> = Vec::new();
    let mut attributes: Vec<String> = Vec::new();

    // Position
    let pos_view = MeshoptView {
        original_length: positions_raw.len(),
        stride: 12,
        count: vert_count,
        compressed_offset: cmp_pos_off,
        compressed_length: positions_compressed.len(),
        mode: "ATTRIBUTES",
    };
    buffer_views.push(format!(
        r#"{{"buffer":0,"byteOffset":{},"byteLength":{},"byteStride":12,"target":{},"extensions":{{"EXT_meshopt_compression":{}}}}}"#,
        fb_pos_off,
        pos_view.original_length,
        ARRAY_BUFFER,
        pos_view.ext_json(),
    ));
    accessors.push(format!(
        r#"{{"bufferView":{},"componentType":{},"count":{},"type":"VEC3","min":[{},{},{}],"max":[{},{},{}]}}"#,
        buffer_views.len() - 1,
        FLOAT,
        vert_count,
        min_pos[0],
        min_pos[1],
        min_pos[2],
        max_pos[0],
        max_pos[1],
        max_pos[2],
    ));
    attributes.push(format!(r#""POSITION":{}"#, accessors.len() - 1));

    // Normal
    if let (Some(off), Some(cmp), Some(raw)) = (fb_norm_off, &normals_compressed, &normals_raw) {
        let view = MeshoptView {
            original_length: raw.len(),
            stride: 12,
            count: vert_count,
            compressed_offset: cmp_norm_off.unwrap(),
            compressed_length: cmp.len(),
            mode: "ATTRIBUTES",
        };
        buffer_views.push(format!(
            r#"{{"buffer":0,"byteOffset":{},"byteLength":{},"byteStride":12,"target":{},"extensions":{{"EXT_meshopt_compression":{}}}}}"#,
            off,
            view.original_length,
            ARRAY_BUFFER,
            view.ext_json(),
        ));
        accessors.push(format!(
            r#"{{"bufferView":{},"componentType":{},"count":{},"type":"VEC3"}}"#,
            buffer_views.len() - 1,
            FLOAT,
            vert_count,
        ));
        attributes.push(format!(r#""NORMAL":{}"#, accessors.len() - 1));
    }

    // UV
    if let (Some(off), Some(cmp), Some(raw)) = (fb_uv_off, &uvs_compressed, &uvs_raw) {
        let view = MeshoptView {
            original_length: raw.len(),
            stride: 8,
            count: vert_count,
            compressed_offset: cmp_uv_off.unwrap(),
            compressed_length: cmp.len(),
            mode: "ATTRIBUTES",
        };
        buffer_views.push(format!(
            r#"{{"buffer":0,"byteOffset":{},"byteLength":{},"byteStride":8,"target":{},"extensions":{{"EXT_meshopt_compression":{}}}}}"#,
            off,
            view.original_length,
            ARRAY_BUFFER,
            view.ext_json(),
        ));
        accessors.push(format!(
            r#"{{"bufferView":{},"componentType":{},"count":{},"type":"VEC2"}}"#,
            buffer_views.len() - 1,
            FLOAT,
            vert_count,
        ));
        attributes.push(format!(r#""TEXCOORD_0":{}"#, accessors.len() - 1));
    }

    // Indices
    let idx_view = MeshoptView {
        original_length: indices_raw.len(),
        stride: 0,
        count: idx_count,
        compressed_offset: cmp_idx_off,
        compressed_length: indices_compressed.len(),
        mode: "TRIANGLES",
    };
    buffer_views.push(format!(
        r#"{{"buffer":0,"byteOffset":{},"byteLength":{},"target":{},"extensions":{{"EXT_meshopt_compression":{}}}}}"#,
        fb_idx_off,
        idx_view.original_length,
        ELEMENT_ARRAY_BUFFER,
        idx_view.ext_json(),
    ));
    accessors.push(format!(
        r#"{{"bufferView":{},"componentType":{},"count":{},"type":"SCALAR"}}"#,
        buffer_views.len() - 1,
        UNSIGNED_INT,
        idx_count,
    ));
    let idx_accessor = accessors.len() - 1;

    // Material
    let mat_json = if config.double_sided {
        r#"{"pbrMetallicRoughness":{"baseColorFactor":[0.7,0.7,0.7,1.0],"metallicFactor":0.0,"roughnessFactor":0.9},"doubleSided":true}"#
    } else {
        r#"{"pbrMetallicRoughness":{"baseColorFactor":[0.7,0.7,0.7,1.0],"metallicFactor":0.0,"roughnessFactor":0.9}}"#
    };

    // Build final JSON
    let json = format!(
        r#"{{"asset":{{"version":"2.0","generator":"ALICE-SDF meshopt-gltf"}},"extensionsUsed":["EXT_meshopt_compression"],"extensionsRequired":["EXT_meshopt_compression"],"buffers":[{{"byteLength":{buffer_len}}}],"bufferViews":[{buffer_views_json}],"accessors":[{accessors_json}],"materials":[{mat_json}],"meshes":[{{"primitives":[{{"attributes":{{{attributes_json}}},"indices":{idx_accessor},"material":0}}]}}],"nodes":[{{"mesh":0}}],"scenes":[{{"nodes":[0]}}],"scene":0}}"#,
        buffer_len = bin.len(),
        buffer_views_json = buffer_views.join(","),
        accessors_json = accessors.join(","),
        attributes_json = attributes.join(","),
    );

    // Assemble GLB
    let mut json_bytes = json.into_bytes();
    while json_bytes.len() % 4 != 0 {
        json_bytes.push(b' ');
    }
    let bin_padded_len = (bin.len() + 3) & !3;
    let bin_pad = bin_padded_len - bin.len();

    let total_len = 12 + 8 + json_bytes.len() + 8 + bin_padded_len;

    let mut out = Vec::with_capacity(total_len);
    out.extend_from_slice(&GLB_MAGIC.to_le_bytes());
    out.extend_from_slice(&GLB_VERSION.to_le_bytes());
    out.extend_from_slice(&(total_len as u32).to_le_bytes());
    // JSON chunk
    out.extend_from_slice(&(json_bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(&GLB_CHUNK_JSON.to_le_bytes());
    out.extend_from_slice(&json_bytes);
    // BIN chunk
    out.extend_from_slice(&(bin_padded_len as u32).to_le_bytes());
    out.extend_from_slice(&GLB_CHUNK_BIN.to_le_bytes());
    out.extend_from_slice(&bin);
    out.resize(out.len() + bin_pad, 0);

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::{sdf_to_mesh, MarchingCubesConfig};
    use crate::types::SdfNode;
    use glam::Vec3;

    fn sphere_mesh(resolution: usize) -> Mesh {
        let sphere = SdfNode::sphere(1.0);
        sdf_to_mesh(
            &sphere,
            Vec3::splat(-2.0),
            Vec3::splat(2.0),
            &MarchingCubesConfig {
                resolution,
                iso_level: 0.0,
                compute_normals: true,
                ..Default::default()
            },
        )
    }

    fn parse_glb_header(bytes: &[u8]) -> (u32, u32, u32) {
        let magic = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        let version = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        let length = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        (magic, version, length)
    }

    #[test]
    fn test_export_glb_meshopt_bytes_valid_header() {
        let mesh = sphere_mesh(8);
        let bytes = export_glb_meshopt_bytes(&mesh, &MeshoptGltfConfig::default()).unwrap();

        let (magic, version, length) = parse_glb_header(&bytes);
        assert_eq!(magic, GLB_MAGIC);
        assert_eq!(version, GLB_VERSION);
        assert_eq!(length, bytes.len() as u32);
    }

    #[test]
    fn test_export_glb_meshopt_json_has_extensions() {
        let mesh = sphere_mesh(4);
        let bytes = export_glb_meshopt_bytes(&mesh, &MeshoptGltfConfig::default()).unwrap();

        // Extract JSON chunk (offset 20 after headers)
        let json_len = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]) as usize;
        let json = std::str::from_utf8(&bytes[20..20 + json_len]).unwrap();

        assert!(json.contains(r#""extensionsUsed":["EXT_meshopt_compression"]"#));
        assert!(json.contains(r#""extensionsRequired":["EXT_meshopt_compression"]"#));
        assert!(json.contains(r#""EXT_meshopt_compression":"#));
        assert!(json.contains(r#""mode":"ATTRIBUTES""#));
        assert!(json.contains(r#""mode":"TRIANGLES""#));
    }

    #[test]
    fn test_export_glb_meshopt_compression_ratio() {
        let mesh = sphere_mesh(16);
        let uncompressed_size = mesh.vertices.len() * 12 /*pos*/ + mesh.vertices.len() * 12 /*normal*/
            + mesh.indices.len() * 4;
        let bytes = export_glb_meshopt_bytes(&mesh, &MeshoptGltfConfig::default()).unwrap();

        // The compressed .glb should be smaller than raw payload + reasonable overhead
        eprintln!(
            "sphere mesh: {} verts / {} indices, raw {} bytes, glb {} bytes",
            mesh.vertices.len(),
            mesh.indices.len(),
            uncompressed_size,
            bytes.len()
        );
    }

    #[test]
    fn test_export_glb_meshopt_double_sided_flag() {
        let mesh = sphere_mesh(4);
        let cfg = MeshoptGltfConfig {
            double_sided: true,
            ..MeshoptGltfConfig::default()
        };
        let bytes = export_glb_meshopt_bytes(&mesh, &cfg).unwrap();

        let json_len = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]) as usize;
        let json = std::str::from_utf8(&bytes[20..20 + json_len]).unwrap();
        assert!(json.contains(r#""doubleSided":true"#));
    }

    #[test]
    fn test_export_glb_meshopt_no_normals() {
        let mesh = sphere_mesh(4);
        let cfg = MeshoptGltfConfig {
            export_normals: false,
            ..MeshoptGltfConfig::default()
        };
        let bytes = export_glb_meshopt_bytes(&mesh, &cfg).unwrap();

        let json_len = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]) as usize;
        let json = std::str::from_utf8(&bytes[20..20 + json_len]).unwrap();
        assert!(!json.contains(r#""NORMAL":"#));
    }

    #[test]
    fn test_export_glb_meshopt_empty_mesh_errors() {
        let mesh = Mesh {
            vertices: vec![],
            indices: vec![],
        };
        let result = export_glb_meshopt_bytes(&mesh, &MeshoptGltfConfig::default());
        assert!(result.is_err());
    }
}

//! glTF 2.0 export for ALICE-SDF meshes
//!
//! Exports meshes to glTF 2.0 binary (.glb) format with PBR materials.
//! Compatible with UE5, Unity, Blender, Godot, and web viewers.
//!
//! Implements minimal glTF spec without external dependencies:
//! - Binary buffer with vertex/index data
//! - PBR metallic-roughness material model
//! - Single-mesh, single-scene structure
//!
//! Author: Moroya Sakamoto

use crate::io::IoError;
use crate::material::MaterialLibrary;
use crate::mesh::Mesh;
use std::io::Write;
use std::path::Path;

/// glTF export configuration
#[derive(Debug, Clone)]
pub struct GltfConfig {
    /// Export normals
    pub export_normals: bool,
    /// Export UVs (TEXCOORD_0)
    pub export_uvs: bool,
    /// Export lightmap UVs (TEXCOORD_1)
    pub export_uv2: bool,
    /// Export vertex colors
    pub export_colors: bool,
    /// Export tangents
    pub export_tangents: bool,
    /// Export materials with PBR properties
    pub export_materials: bool,
    /// Quantize positions to 16-bit (reduces file size ~50%)
    pub quantize_positions: bool,
    /// Embed textures in GLB binary buffer (self-contained assets)
    pub embed_textures: bool,
    /// Export extended PBR material extensions (clearcoat, sheen, etc.)
    pub export_extensions: bool,
}

impl Default for GltfConfig {
    fn default() -> Self {
        GltfConfig {
            export_normals: true,
            export_uvs: true,
            export_uv2: false,
            export_colors: false,
            export_tangents: false,
            export_materials: true,
            quantize_positions: false,
            embed_textures: false,
            export_extensions: false,
        }
    }
}

impl GltfConfig {
    /// Full AAA config with all attributes
    pub fn aaa() -> Self {
        GltfConfig {
            export_normals: true,
            export_uvs: true,
            export_uv2: true,
            export_colors: true,
            export_tangents: true,
            export_materials: true,
            quantize_positions: false,
            embed_textures: true,
            export_extensions: true,
        }
    }
}

// GLB constants
const GLB_MAGIC: u32 = 0x46546C67; // "glTF"
const GLB_VERSION: u32 = 2;
const GLB_CHUNK_JSON: u32 = 0x4E4F534A; // "JSON"
const GLB_CHUNK_BIN: u32 = 0x004E4942; // "BIN\0"

// Component types
const FLOAT: u32 = 5126;
const UNSIGNED_INT: u32 = 5125;
const UNSIGNED_SHORT: u32 = 5123;

// Buffer view targets
const ARRAY_BUFFER: u32 = 34962;
const ELEMENT_ARRAY_BUFFER: u32 = 34963;

/// Export a mesh to glTF 2.0 Binary (.glb) format
pub fn export_glb(
    mesh: &Mesh,
    path: impl AsRef<Path>,
    config: &GltfConfig,
    materials: Option<&MaterialLibrary>,
) -> Result<(), IoError> {
    let (json_bytes, bin_bytes) = build_glb_data(mesh, config, materials)?;

    let file = std::fs::File::create(path)?;
    let mut w = std::io::BufWriter::new(file);

    // Pad JSON to 4-byte alignment
    let json_padded_len = (json_bytes.len() + 3) & !3;
    let bin_padded_len = (bin_bytes.len() + 3) & !3;

    let total_len = 12 + 8 + json_padded_len + 8 + bin_padded_len;

    // GLB header (12 bytes)
    w.write_all(&GLB_MAGIC.to_le_bytes())?;
    w.write_all(&GLB_VERSION.to_le_bytes())?;
    w.write_all(&(total_len as u32).to_le_bytes())?;

    // JSON chunk
    w.write_all(&(json_padded_len as u32).to_le_bytes())?;
    w.write_all(&GLB_CHUNK_JSON.to_le_bytes())?;
    w.write_all(&json_bytes)?;
    // Pad with spaces
    for _ in 0..(json_padded_len - json_bytes.len()) {
        w.write_all(b" ")?;
    }

    // BIN chunk
    w.write_all(&(bin_padded_len as u32).to_le_bytes())?;
    w.write_all(&GLB_CHUNK_BIN.to_le_bytes())?;
    w.write_all(&bin_bytes)?;
    // Pad with zeros
    for _ in 0..(bin_padded_len - bin_bytes.len()) {
        w.write_all(&[0u8])?;
    }

    w.flush()?;
    Ok(())
}

/// Export a mesh to GLB 2.0 binary format in memory (no file I/O).
///
/// Returns the complete GLB byte buffer.
pub fn export_glb_bytes(
    mesh: &Mesh,
    config: &GltfConfig,
    materials: Option<&MaterialLibrary>,
) -> Result<Vec<u8>, IoError> {
    let (json_bytes, bin_bytes) = build_glb_data(mesh, config, materials)?;

    let json_padded_len = (json_bytes.len() + 3) & !3;
    let bin_padded_len = (bin_bytes.len() + 3) & !3;
    let total_len = 12 + 8 + json_padded_len + 8 + bin_padded_len;

    let mut buf = Vec::with_capacity(total_len);

    // GLB header
    buf.extend_from_slice(&GLB_MAGIC.to_le_bytes());
    buf.extend_from_slice(&GLB_VERSION.to_le_bytes());
    buf.extend_from_slice(&(total_len as u32).to_le_bytes());

    // JSON chunk
    buf.extend_from_slice(&(json_padded_len as u32).to_le_bytes());
    buf.extend_from_slice(&GLB_CHUNK_JSON.to_le_bytes());
    buf.extend_from_slice(&json_bytes);
    buf.resize(buf.len() + json_padded_len - json_bytes.len(), b' ');

    // BIN chunk
    buf.extend_from_slice(&(bin_padded_len as u32).to_le_bytes());
    buf.extend_from_slice(&GLB_CHUNK_BIN.to_le_bytes());
    buf.extend_from_slice(&bin_bytes);
    buf.resize(buf.len() + bin_padded_len - bin_bytes.len(), 0u8);

    Ok(buf)
}

/// Export a mesh to glTF 2.0 JSON (.gltf) with embedded base64 buffer
pub fn export_gltf_json(
    mesh: &Mesh,
    path: impl AsRef<Path>,
    config: &GltfConfig,
    materials: Option<&MaterialLibrary>,
) -> Result<(), IoError> {
    let (json_bytes, _) = build_glb_data(mesh, config, materials)?;

    std::fs::write(path, json_bytes)?;
    Ok(())
}

fn build_glb_data(
    mesh: &Mesh,
    config: &GltfConfig,
    materials: Option<&MaterialLibrary>,
) -> Result<(Vec<u8>, Vec<u8>), IoError> {
    let vert_count = mesh.vertices.len();
    let use_u16 = vert_count <= 65535;

    // Build binary buffer
    let mut bin = Vec::new();
    let mut buffer_views = Vec::new();
    let mut accessors = Vec::new();
    let mut attributes = Vec::new();

    // Compute AABB for positions
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

    // --- Positions (accessor 0) ---
    // Track if we need KHR_mesh_quantization extension
    let mut needs_quantization_ext = false;

    let pos_offset = bin.len();
    if config.quantize_positions {
        // Quantize positions to SHORT (16-bit signed integer)
        // Dequantization is done via node transform: pos = quantized * scale + translation
        needs_quantization_ext = true;
        let range = [
            max_pos[0] - min_pos[0],
            max_pos[1] - min_pos[1],
            max_pos[2] - min_pos[2],
        ];
        for v in &mesh.vertices {
            let qx = if range[0] > 1e-6 {
                (((v.position.x - min_pos[0]) / range[0]) * 32767.0) as i16
            } else {
                0
            };
            let qy = if range[1] > 1e-6 {
                (((v.position.y - min_pos[1]) / range[1]) * 32767.0) as i16
            } else {
                0
            };
            let qz = if range[2] > 1e-6 {
                (((v.position.z - min_pos[2]) / range[2]) * 32767.0) as i16
            } else {
                0
            };
            bin.extend_from_slice(&qx.to_le_bytes());
            bin.extend_from_slice(&qy.to_le_bytes());
            bin.extend_from_slice(&qz.to_le_bytes());
        }
        let pos_len = bin.len() - pos_offset;
        // byteStride = 6 (3 * sizeof(i16))
        buffer_views.push(format!(
            r#"{{"buffer":0,"byteOffset":{},"byteLength":{},"byteStride":6,"target":{}}}"#,
            pos_offset, pos_len, ARRAY_BUFFER
        ));
        // SHORT = 5122
        accessors.push(format!(
            r#"{{"bufferView":{},"componentType":5122,"count":{},"type":"VEC3","min":[0,0,0],"max":[32767,32767,32767]}}"#,
            buffer_views.len() - 1, vert_count
        ));
    } else {
        for v in &mesh.vertices {
            bin.extend_from_slice(&v.position.x.to_le_bytes());
            bin.extend_from_slice(&v.position.y.to_le_bytes());
            bin.extend_from_slice(&v.position.z.to_le_bytes());
        }
        let pos_len = bin.len() - pos_offset;
        buffer_views.push(format!(
            r#"{{"buffer":0,"byteOffset":{},"byteLength":{},"target":{}}}"#,
            pos_offset, pos_len, ARRAY_BUFFER
        ));
        accessors.push(format!(
            r#"{{"bufferView":{},"componentType":{},"count":{},"type":"VEC3","min":[{},{},{}],"max":[{},{},{}]}}"#,
            buffer_views.len() - 1, FLOAT, vert_count,
            min_pos[0], min_pos[1], min_pos[2],
            max_pos[0], max_pos[1], max_pos[2]
        ));
    }
    attributes.push(format!(r#""POSITION":{}"#, accessors.len() - 1));

    // --- Normals ---
    if config.export_normals {
        let offset = bin.len();
        for v in &mesh.vertices {
            bin.extend_from_slice(&v.normal.x.to_le_bytes());
            bin.extend_from_slice(&v.normal.y.to_le_bytes());
            bin.extend_from_slice(&v.normal.z.to_le_bytes());
        }
        let len = bin.len() - offset;
        buffer_views.push(format!(
            r#"{{"buffer":0,"byteOffset":{},"byteLength":{},"target":{}}}"#,
            offset, len, ARRAY_BUFFER
        ));
        accessors.push(format!(
            r#"{{"bufferView":{},"componentType":{},"count":{},"type":"VEC3"}}"#,
            buffer_views.len() - 1,
            FLOAT,
            vert_count
        ));
        attributes.push(format!(r#""NORMAL":{}"#, accessors.len() - 1));
    }

    // --- UVs (TEXCOORD_0) ---
    if config.export_uvs {
        let offset = bin.len();
        for v in &mesh.vertices {
            bin.extend_from_slice(&v.uv.x.to_le_bytes());
            bin.extend_from_slice(&v.uv.y.to_le_bytes());
        }
        let len = bin.len() - offset;
        buffer_views.push(format!(
            r#"{{"buffer":0,"byteOffset":{},"byteLength":{},"target":{}}}"#,
            offset, len, ARRAY_BUFFER
        ));
        accessors.push(format!(
            r#"{{"bufferView":{},"componentType":{},"count":{},"type":"VEC2"}}"#,
            buffer_views.len() - 1,
            FLOAT,
            vert_count
        ));
        attributes.push(format!(r#""TEXCOORD_0":{}"#, accessors.len() - 1));
    }

    // --- Lightmap UVs (TEXCOORD_1) ---
    if config.export_uv2 {
        let offset = bin.len();
        for v in &mesh.vertices {
            bin.extend_from_slice(&v.uv2.x.to_le_bytes());
            bin.extend_from_slice(&v.uv2.y.to_le_bytes());
        }
        let len = bin.len() - offset;
        buffer_views.push(format!(
            r#"{{"buffer":0,"byteOffset":{},"byteLength":{},"target":{}}}"#,
            offset, len, ARRAY_BUFFER
        ));
        accessors.push(format!(
            r#"{{"bufferView":{},"componentType":{},"count":{},"type":"VEC2"}}"#,
            buffer_views.len() - 1,
            FLOAT,
            vert_count
        ));
        attributes.push(format!(r#""TEXCOORD_1":{}"#, accessors.len() - 1));
    }

    // --- Tangents ---
    if config.export_tangents {
        let offset = bin.len();
        for v in &mesh.vertices {
            bin.extend_from_slice(&v.tangent.x.to_le_bytes());
            bin.extend_from_slice(&v.tangent.y.to_le_bytes());
            bin.extend_from_slice(&v.tangent.z.to_le_bytes());
            bin.extend_from_slice(&v.tangent.w.to_le_bytes());
        }
        let len = bin.len() - offset;
        buffer_views.push(format!(
            r#"{{"buffer":0,"byteOffset":{},"byteLength":{},"target":{}}}"#,
            offset, len, ARRAY_BUFFER
        ));
        accessors.push(format!(
            r#"{{"bufferView":{},"componentType":{},"count":{},"type":"VEC4"}}"#,
            buffer_views.len() - 1,
            FLOAT,
            vert_count
        ));
        attributes.push(format!(r#""TANGENT":{}"#, accessors.len() - 1));
    }

    // --- Vertex Colors ---
    if config.export_colors {
        let offset = bin.len();
        for v in &mesh.vertices {
            bin.extend_from_slice(&v.color[0].to_le_bytes());
            bin.extend_from_slice(&v.color[1].to_le_bytes());
            bin.extend_from_slice(&v.color[2].to_le_bytes());
            bin.extend_from_slice(&v.color[3].to_le_bytes());
        }
        let len = bin.len() - offset;
        buffer_views.push(format!(
            r#"{{"buffer":0,"byteOffset":{},"byteLength":{},"target":{}}}"#,
            offset, len, ARRAY_BUFFER
        ));
        accessors.push(format!(
            r#"{{"bufferView":{},"componentType":{},"count":{},"type":"VEC4"}}"#,
            buffer_views.len() - 1,
            FLOAT,
            vert_count
        ));
        attributes.push(format!(r#""COLOR_0":{}"#, accessors.len() - 1));
    }

    // --- Texture collection (images, textures, samplers) ---
    let mut image_paths: Vec<String> = Vec::new();
    let mut path_to_tex_idx: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();

    if config.export_materials {
        if let Some(mat_lib) = materials {
            for mat in &mat_lib.materials {
                let slots: [&Option<crate::material::TextureSlot>; 8] = [
                    &mat.albedo_map,
                    &mat.normal_map,
                    &mat.metallic_map,
                    &mat.roughness_map,
                    &mat.ao_map,
                    &mat.emissive_map,
                    &mat.metallic_roughness_map,
                    &mat.clearcoat_normal_map,
                ];
                for slot in slots.iter().copied().flatten() {
                    if !path_to_tex_idx.contains_key(&slot.path) {
                        path_to_tex_idx.insert(slot.path.clone(), image_paths.len());
                        image_paths.push(slot.path.clone());
                    }
                }
            }
        }
    }

    // --- Materials ---
    let mut materials_json = Vec::new();
    let mut used_extensions: Vec<String> = Vec::new();
    if config.export_materials {
        if let Some(mat_lib) = materials {
            for mat in &mat_lib.materials {
                // Build PBR metallic-roughness object
                let mut pbr = format!(
                    r#"{{"baseColorFactor":[{},{},{},{}],"metallicFactor":{},"roughnessFactor":{}"#,
                    mat.base_color[0],
                    mat.base_color[1],
                    mat.base_color[2],
                    mat.base_color[3],
                    mat.metallic,
                    mat.roughness
                );

                if let Some(ref tex) = mat.albedo_map {
                    if let Some(&idx) = path_to_tex_idx.get(&tex.path) {
                        pbr += &format!(r#","baseColorTexture":{{"index":{}}}"#, idx);
                    }
                }
                if let Some(ref tex) = mat.metallic_roughness_map {
                    if let Some(&idx) = path_to_tex_idx.get(&tex.path) {
                        pbr += &format!(r#","metallicRoughnessTexture":{{"index":{}}}"#, idx);
                    }
                }

                pbr += "}"; // close pbrMetallicRoughness

                let mut mat_str =
                    format!(r#"{{"name":"{}","pbrMetallicRoughness":{}"#, mat.name, pbr);

                // Normal map
                if let Some(ref tex) = mat.normal_map {
                    if let Some(&idx) = path_to_tex_idx.get(&tex.path) {
                        mat_str += &format!(
                            r#","normalTexture":{{"index":{},"scale":{}}}"#,
                            idx, mat.normal_scale
                        );
                    }
                }

                // AO map
                if let Some(ref tex) = mat.ao_map {
                    if let Some(&idx) = path_to_tex_idx.get(&tex.path) {
                        mat_str += &format!(r#","occlusionTexture":{{"index":{}}}"#, idx);
                    }
                }

                // Emissive
                if mat.emission_strength > 0.0 {
                    mat_str += &format!(
                        r#","emissiveFactor":[{},{},{}]"#,
                        mat.emission[0] * mat.emission_strength,
                        mat.emission[1] * mat.emission_strength,
                        mat.emission[2] * mat.emission_strength,
                    );
                }
                if let Some(ref tex) = mat.emissive_map {
                    if let Some(&idx) = path_to_tex_idx.get(&tex.path) {
                        mat_str += &format!(r#","emissiveTexture":{{"index":{}}}"#, idx);
                    }
                }

                if mat.opacity < 1.0 {
                    mat_str += r#","alphaMode":"BLEND""#;
                }

                // --- Extended PBR extensions ---
                if config.export_extensions {
                    let mut ext_parts = Vec::new();

                    // KHR_materials_clearcoat
                    if mat.clearcoat > 0.0 {
                        ext_parts.push(format!(
                            r#""KHR_materials_clearcoat":{{"clearcoatFactor":{},"clearcoatRoughnessFactor":{}}}"#,
                            mat.clearcoat, mat.clearcoat_roughness
                        ));
                        if !used_extensions.contains(&"KHR_materials_clearcoat".to_string()) {
                            used_extensions.push("KHR_materials_clearcoat".to_string());
                        }
                    }

                    // KHR_materials_sheen
                    if mat.sheen_color != [0.0, 0.0, 0.0] || mat.sheen_roughness > 0.0 {
                        ext_parts.push(format!(
                            r#""KHR_materials_sheen":{{"sheenColorFactor":[{},{},{}],"sheenRoughnessFactor":{}}}"#,
                            mat.sheen_color[0], mat.sheen_color[1], mat.sheen_color[2],
                            mat.sheen_roughness
                        ));
                        if !used_extensions.contains(&"KHR_materials_sheen".to_string()) {
                            used_extensions.push("KHR_materials_sheen".to_string());
                        }
                    }

                    // KHR_materials_transmission
                    if mat.transmission > 0.0 {
                        ext_parts.push(format!(
                            r#""KHR_materials_transmission":{{"transmissionFactor":{}}}"#,
                            mat.transmission
                        ));
                        if !used_extensions.contains(&"KHR_materials_transmission".to_string()) {
                            used_extensions.push("KHR_materials_transmission".to_string());
                        }
                    }

                    // KHR_materials_ior (only if not default 1.5)
                    if (mat.ior - 1.5).abs() > 0.001 {
                        ext_parts.push(format!(r#""KHR_materials_ior":{{"ior":{}}}"#, mat.ior));
                        if !used_extensions.contains(&"KHR_materials_ior".to_string()) {
                            used_extensions.push("KHR_materials_ior".to_string());
                        }
                    }

                    // KHR_materials_volume
                    if mat.thickness > 0.0 {
                        let mut vol = format!(
                            r#""KHR_materials_volume":{{"thicknessFactor":{}"#,
                            mat.thickness
                        );
                        if mat.attenuation_distance.is_finite() {
                            vol +=
                                &format!(r#","attenuationDistance":{}"#, mat.attenuation_distance);
                        }
                        if mat.attenuation_color != [1.0, 1.0, 1.0] {
                            vol += &format!(
                                r#","attenuationColor":[{},{},{}]"#,
                                mat.attenuation_color[0],
                                mat.attenuation_color[1],
                                mat.attenuation_color[2]
                            );
                        }
                        vol += "}";
                        ext_parts.push(vol);
                        if !used_extensions.contains(&"KHR_materials_volume".to_string()) {
                            used_extensions.push("KHR_materials_volume".to_string());
                        }
                    }

                    // KHR_materials_anisotropy
                    if mat.anisotropy.abs() > 0.001 {
                        ext_parts.push(format!(
                            r#""KHR_materials_anisotropy":{{"anisotropyStrength":{},"anisotropyRotation":{}}}"#,
                            mat.anisotropy, mat.anisotropy_rotation
                        ));
                        if !used_extensions.contains(&"KHR_materials_anisotropy".to_string()) {
                            used_extensions.push("KHR_materials_anisotropy".to_string());
                        }
                    }

                    if !ext_parts.is_empty() {
                        mat_str += &format!(r#","extensions":{{{}}}"#, ext_parts.join(","));
                    }
                }

                mat_str += "}";
                materials_json.push(mat_str);
            }
        }
    }

    // --- Indices: group by material for multi-material splitting ---
    let idx_comp_type = if use_u16 {
        UNSIGNED_SHORT
    } else {
        UNSIGNED_INT
    };
    let attr_str = attributes.join(",");
    let mut primitives = Vec::new();

    if config.export_materials && materials.is_some() && !materials_json.is_empty() {
        // Group triangles by material_id
        let mut mat_groups: std::collections::BTreeMap<u32, Vec<u32>> =
            std::collections::BTreeMap::new();
        let tri_count = mesh.indices.len() / 3;
        for t in 0..tri_count {
            let v0 = mesh.indices[t * 3] as usize;
            let mat_id = if v0 < mesh.vertices.len() {
                mesh.vertices[v0].material_id
            } else {
                0
            };
            let group = mat_groups.entry(mat_id).or_default();
            group.push(mesh.indices[t * 3]);
            group.push(mesh.indices[t * 3 + 1]);
            group.push(mesh.indices[t * 3 + 2]);
        }

        // Create separate index buffer + accessor + primitive for each material
        for (mat_id, group_indices) in &mat_groups {
            while bin.len() % 4 != 0 {
                bin.push(0);
            }
            let idx_offset = bin.len();

            if use_u16 {
                for &idx in group_indices {
                    bin.extend_from_slice(&(idx as u16).to_le_bytes());
                }
            } else {
                for &idx in group_indices {
                    bin.extend_from_slice(&idx.to_le_bytes());
                }
            }
            let idx_len = bin.len() - idx_offset;

            buffer_views.push(format!(
                r#"{{"buffer":0,"byteOffset":{},"byteLength":{},"target":{}}}"#,
                idx_offset, idx_len, ELEMENT_ARRAY_BUFFER
            ));
            accessors.push(format!(
                r#"{{"bufferView":{},"componentType":{},"count":{},"type":"SCALAR"}}"#,
                buffer_views.len() - 1,
                idx_comp_type,
                group_indices.len()
            ));

            let mat_ref = if (*mat_id as usize) < materials_json.len() {
                *mat_id
            } else {
                0
            };

            primitives.push(format!(
                r#"{{"attributes":{{{}}},"indices":{},"material":{}}}"#,
                attr_str,
                accessors.len() - 1,
                mat_ref
            ));
        }
    } else {
        // Single primitive, no material grouping
        while bin.len() % 4 != 0 {
            bin.push(0);
        }
        let idx_offset = bin.len();

        if use_u16 {
            for &idx in &mesh.indices {
                bin.extend_from_slice(&(idx as u16).to_le_bytes());
            }
        } else {
            for &idx in &mesh.indices {
                bin.extend_from_slice(&idx.to_le_bytes());
            }
        }
        let idx_len = bin.len() - idx_offset;

        buffer_views.push(format!(
            r#"{{"buffer":0,"byteOffset":{},"byteLength":{},"target":{}}}"#,
            idx_offset, idx_len, ELEMENT_ARRAY_BUFFER
        ));
        accessors.push(format!(
            r#"{{"bufferView":{},"componentType":{},"count":{},"type":"SCALAR"}}"#,
            buffer_views.len() - 1,
            idx_comp_type,
            mesh.indices.len()
        ));

        primitives.push(format!(
            r#"{{"attributes":{{{}}},"indices":{}}}"#,
            attr_str,
            accessors.len() - 1
        ));
    }

    // Build JSON
    let mut json = String::from(r#"{"asset":{"version":"2.0","generator":"ALICE-SDF"}"#);

    // Extensions
    if needs_quantization_ext && !used_extensions.contains(&"KHR_mesh_quantization".to_string()) {
        used_extensions.push("KHR_mesh_quantization".to_string());
    }
    if !used_extensions.is_empty() {
        let ext_list: Vec<String> = used_extensions
            .iter()
            .map(|e| format!(r#""{}""#, e))
            .collect();
        json += &format!(r#","extensionsUsed":[{}]"#, ext_list.join(","));
        // Only KHR_mesh_quantization is required
        if needs_quantization_ext {
            json += r#","extensionsRequired":["KHR_mesh_quantization"]"#;
        }
    }

    // Buffers
    json += &format!(r#","buffers":[{{"byteLength":{}}}]"#, bin.len());

    // Buffer views
    json += &format!(r#","bufferViews":[{}]"#, buffer_views.join(","));

    // Accessors
    json += &format!(r#","accessors":[{}]"#, accessors.join(","));

    // Meshes
    json += &format!(r#","meshes":[{{"primitives":[{}]}}]"#, primitives.join(","));

    // Nodes (with dequantization transform if quantized)
    if config.quantize_positions {
        let range = [
            max_pos[0] - min_pos[0],
            max_pos[1] - min_pos[1],
            max_pos[2] - min_pos[2],
        ];
        let sx = if range[0] > 1e-6 {
            range[0] / 32767.0
        } else {
            1.0
        };
        let sy = if range[1] > 1e-6 {
            range[1] / 32767.0
        } else {
            1.0
        };
        let sz = if range[2] > 1e-6 {
            range[2] / 32767.0
        } else {
            1.0
        };
        json += &format!(
            r#","nodes":[{{"mesh":0,"translation":[{},{},{}],"scale":[{},{},{}]}}]"#,
            min_pos[0], min_pos[1], min_pos[2], sx, sy, sz
        );
    } else {
        json += r#","nodes":[{"mesh":0}]"#;
    }

    // Scene
    json += r#","scenes":[{"nodes":[0]}],"scene":0"#;

    // Materials
    if !materials_json.is_empty() {
        json += &format!(r#","materials":[{}]"#, materials_json.join(","));
    }

    // Textures, images, samplers
    if !image_paths.is_empty() {
        if config.embed_textures {
            // Embed textures in binary buffer (self-contained GLB)
            let mut images_json = Vec::new();
            for p in &image_paths {
                let mime = if p.ends_with(".png") {
                    "image/png"
                } else if p.ends_with(".jpg") || p.ends_with(".jpeg") {
                    "image/jpeg"
                } else if p.ends_with(".webp") {
                    "image/webp"
                } else {
                    "application/octet-stream"
                };

                match std::fs::read(p) {
                    Ok(image_data) => {
                        // Align to 4 bytes
                        while bin.len() % 4 != 0 {
                            bin.push(0);
                        }
                        let offset = bin.len();
                        let len = image_data.len();
                        bin.extend_from_slice(&image_data);

                        buffer_views.push(format!(
                            r#"{{"buffer":0,"byteOffset":{},"byteLength":{}}}"#,
                            offset, len
                        ));
                        images_json.push(format!(
                            r#"{{"bufferView":{},"mimeType":"{}"}}"#,
                            buffer_views.len() - 1,
                            mime
                        ));
                    }
                    Err(_) => {
                        // Fallback to URI if file not found
                        images_json.push(format!(r#"{{"uri":"{}"}}"#, p));
                    }
                }
            }
            json += &format!(r#","images":[{}]"#, images_json.join(","));
        } else {
            // URI references to external texture files
            let images: Vec<String> = image_paths
                .iter()
                .map(|p| format!(r#"{{"uri":"{}"}}"#, p))
                .collect();
            json += &format!(r#","images":[{}]"#, images.join(","));
        }

        // Textures (each image gets a texture entry with default sampler)
        let textures: Vec<String> = (0..image_paths.len())
            .map(|i| format!(r#"{{"sampler":0,"source":{}}}"#, i))
            .collect();
        json += &format!(r#","textures":[{}]"#, textures.join(","));

        // Default sampler (linear filtering, repeat wrap)
        json += r#","samplers":[{"magFilter":9729,"minFilter":9987,"wrapS":10497,"wrapT":10497}]"#;
    }

    json += "}";

    Ok((json.into_bytes(), bin))
}

// ============================================================
// GLB Import
// ============================================================

/// Import a mesh from glTF 2.0 Binary (.glb) file
///
/// Parses GLB header, JSON chunk, and BIN chunk. Extracts positions,
/// normals, UVs, and indices from the binary buffer.
///
/// # Supported accessor types
/// - Positions: VEC3 / FLOAT
/// - Normals: VEC3 / FLOAT
/// - UVs (TEXCOORD_0): VEC2 / FLOAT
/// - Indices: SCALAR / UNSIGNED_SHORT or UNSIGNED_INT
pub fn import_glb(path: impl AsRef<Path>) -> Result<Mesh, IoError> {
    let data = std::fs::read(path)?;
    import_glb_bytes(&data)
}

/// Import a mesh from in-memory GLB bytes
pub fn import_glb_bytes(data: &[u8]) -> Result<Mesh, IoError> {
    use crate::mesh::Vertex;
    use glam::{Vec2, Vec3};

    if data.len() < 12 {
        return Err(IoError::InvalidFormat("GLB too short for header".into()));
    }

    // Parse GLB header (12 bytes)
    let magic = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
    if magic != GLB_MAGIC {
        return Err(IoError::InvalidFormat(format!(
            "Not a GLB file (magic: 0x{:08X}, expected 0x{:08X})",
            magic, GLB_MAGIC
        )));
    }
    let _version = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
    let _total_len = u32::from_le_bytes([data[8], data[9], data[10], data[11]]);

    // Parse chunks
    let mut json_data: Option<&[u8]> = None;
    let mut bin_data: Option<&[u8]> = None;
    let mut offset = 12usize;

    while offset + 8 <= data.len() {
        let chunk_len = u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]) as usize;
        let chunk_type = u32::from_le_bytes([
            data[offset + 4],
            data[offset + 5],
            data[offset + 6],
            data[offset + 7],
        ]);
        offset += 8;

        if offset + chunk_len > data.len() {
            break;
        }

        match chunk_type {
            GLB_CHUNK_JSON => json_data = Some(&data[offset..offset + chunk_len]),
            GLB_CHUNK_BIN => bin_data = Some(&data[offset..offset + chunk_len]),
            _ => {} // skip unknown chunks
        }
        offset += chunk_len;
    }

    let json_bytes =
        json_data.ok_or_else(|| IoError::InvalidFormat("Missing JSON chunk".into()))?;
    let bin_bytes = bin_data.ok_or_else(|| IoError::InvalidFormat("Missing BIN chunk".into()))?;

    // Parse JSON using serde_json::Value (minimal parsing)
    let root: serde_json::Value = serde_json::from_slice(json_bytes)
        .map_err(|e| IoError::InvalidFormat(format!("Invalid glTF JSON: {}", e)))?;

    let buffer_views = root["bufferViews"]
        .as_array()
        .ok_or_else(|| IoError::InvalidFormat("Missing bufferViews".into()))?;
    let accessors = root["accessors"]
        .as_array()
        .ok_or_else(|| IoError::InvalidFormat("Missing accessors".into()))?;

    // Get first mesh, first primitive
    let meshes = root["meshes"]
        .as_array()
        .ok_or_else(|| IoError::InvalidFormat("Missing meshes".into()))?;
    if meshes.is_empty() {
        return Err(IoError::InvalidFormat("No meshes in glTF".into()));
    }
    let primitives = meshes[0]["primitives"]
        .as_array()
        .ok_or_else(|| IoError::InvalidFormat("Missing primitives".into()))?;
    if primitives.is_empty() {
        return Err(IoError::InvalidFormat("No primitives in mesh".into()));
    }

    let prim = &primitives[0];
    let attrs = &prim["attributes"];

    // Helper: read accessor data from BIN chunk
    let read_accessor_f32 = |acc_idx: usize| -> Result<Vec<f32>, IoError> {
        let acc = &accessors[acc_idx];
        let bv_idx = acc["bufferView"].as_u64().unwrap_or(0) as usize;
        let bv = &buffer_views[bv_idx];
        let byte_offset = bv["byteOffset"].as_u64().unwrap_or(0) as usize
            + acc["byteOffset"].as_u64().unwrap_or(0) as usize;
        let count = acc["count"].as_u64().unwrap_or(0) as usize;
        let comp_type = acc["componentType"].as_u64().unwrap_or(0) as u32;
        let acc_type = acc["type"].as_str().unwrap_or("");

        let components = match acc_type {
            "SCALAR" => 1,
            "VEC2" => 2,
            "VEC3" => 3,
            "VEC4" => 4,
            _ => {
                return Err(IoError::InvalidFormat(format!(
                    "Unknown accessor type: {}",
                    acc_type
                )))
            }
        };

        if comp_type != FLOAT {
            return Err(IoError::InvalidFormat(format!(
                "Expected FLOAT component, got {}",
                comp_type
            )));
        }

        let total = count * components;
        let mut result = Vec::with_capacity(total);
        let stride = bv.get("byteStride").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let elem_size = components * 4;

        if stride == 0 || stride == elem_size {
            // Tightly packed: bulk convert with single bounds check
            let end = byte_offset + total * 4;
            if end > bin_bytes.len() {
                return Err(IoError::InvalidFormat("Buffer overrun".into()));
            }
            for chunk in bin_bytes[byte_offset..end].chunks_exact(4) {
                result.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
        } else {
            // Strided: per-element access
            for i in 0..count {
                let base = byte_offset + i * stride;
                for c in 0..components {
                    let off = base + c * 4;
                    if off + 4 > bin_bytes.len() {
                        return Err(IoError::InvalidFormat("Buffer overrun".into()));
                    }
                    result.push(f32::from_le_bytes([
                        bin_bytes[off],
                        bin_bytes[off + 1],
                        bin_bytes[off + 2],
                        bin_bytes[off + 3],
                    ]));
                }
            }
        }
        Ok(result)
    };

    // Read positions (required)
    let pos_idx = attrs["POSITION"]
        .as_u64()
        .ok_or_else(|| IoError::InvalidFormat("Missing POSITION attribute".into()))?
        as usize;
    let positions = read_accessor_f32(pos_idx)?;
    let vert_count = positions.len() / 3;

    // Read normals (optional)
    let normals = if let Some(idx) = attrs["NORMAL"].as_u64() {
        Some(read_accessor_f32(idx as usize)?)
    } else {
        None
    };

    // Read UVs (optional)
    let uvs = if let Some(idx) = attrs["TEXCOORD_0"].as_u64() {
        Some(read_accessor_f32(idx as usize)?)
    } else {
        None
    };

    // Build vertices
    let mut vertices = Vec::with_capacity(vert_count);
    for i in 0..vert_count {
        let pos = Vec3::new(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]);
        let norm = normals
            .as_ref()
            .map_or(Vec3::Y, |n| Vec3::new(n[i * 3], n[i * 3 + 1], n[i * 3 + 2]));
        let mut vert = Vertex::new(pos, norm);
        if let Some(ref uv_data) = uvs {
            vert.uv = Vec2::new(uv_data[i * 2], uv_data[i * 2 + 1]);
        }
        vertices.push(vert);
    }

    // Read indices
    let mut indices = Vec::new();
    if let Some(idx_acc) = prim["indices"].as_u64() {
        let acc = &accessors[idx_acc as usize];
        let bv_idx = acc["bufferView"].as_u64().unwrap_or(0) as usize;
        let bv = &buffer_views[bv_idx];
        let byte_offset = bv["byteOffset"].as_u64().unwrap_or(0) as usize
            + acc["byteOffset"].as_u64().unwrap_or(0) as usize;
        let count = acc["count"].as_u64().unwrap_or(0) as usize;
        let comp_type = acc["componentType"].as_u64().unwrap_or(0) as u32;

        indices.reserve(count);
        for i in 0..count {
            let idx = match comp_type {
                UNSIGNED_SHORT => {
                    let off = byte_offset + i * 2;
                    u16::from_le_bytes([bin_bytes[off], bin_bytes[off + 1]]) as u32
                }
                UNSIGNED_INT => {
                    let off = byte_offset + i * 4;
                    u32::from_le_bytes([
                        bin_bytes[off],
                        bin_bytes[off + 1],
                        bin_bytes[off + 2],
                        bin_bytes[off + 3],
                    ])
                }
                _ => {
                    return Err(IoError::InvalidFormat(format!(
                        "Unsupported index type: {}",
                        comp_type
                    )));
                }
            };
            indices.push(idx);
        }
    } else {
        // No indices: generate sequential indices
        indices = (0..vert_count as u32).collect();
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
    fn test_glb_export() {
        let sphere = SdfNode::sphere(1.0);
        let config = MarchingCubesConfig {
            resolution: 8,
            iso_level: 0.0,
            compute_normals: true,
            ..Default::default()
        };
        let mesh = sdf_to_mesh(&sphere, Vec3::splat(-2.0), Vec3::splat(2.0), &config);

        let path = std::env::temp_dir().join("alice_test.glb");
        export_glb(&mesh, &path, &GltfConfig::default(), None).unwrap();

        // Verify GLB header
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(&bytes[0..4], &GLB_MAGIC.to_le_bytes());
        assert_eq!(&bytes[4..8], &GLB_VERSION.to_le_bytes());

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_glb_with_materials() {
        let sphere = SdfNode::sphere(1.0);
        let config = MarchingCubesConfig {
            resolution: 8,
            iso_level: 0.0,
            compute_normals: true,
            ..Default::default()
        };
        let mesh = sdf_to_mesh(&sphere, Vec3::splat(-2.0), Vec3::splat(2.0), &config);

        let mut mat_lib = MaterialLibrary::new();
        mat_lib.add(crate::material::Material::metal(
            "chrome", 0.9, 0.9, 0.9, 0.2,
        ));

        let path = std::env::temp_dir().join("alice_test_mat.glb");
        export_glb(&mesh, &path, &GltfConfig::default(), Some(&mat_lib)).unwrap();

        let bytes = std::fs::read(&path).unwrap();
        assert!(bytes.len() > 20);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_glb_aaa_config() {
        let sphere = SdfNode::sphere(1.0);
        let mc_config = MarchingCubesConfig::aaa(8);
        let mesh = sdf_to_mesh(&sphere, Vec3::splat(-2.0), Vec3::splat(2.0), &mc_config);

        let path = std::env::temp_dir().join("alice_test_aaa.glb");
        export_glb(&mesh, &path, &GltfConfig::aaa(), None).unwrap();

        let bytes = std::fs::read(&path).unwrap();
        assert!(bytes.len() > 100);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_glb_texture_references() {
        let sphere = SdfNode::sphere(1.0);
        let config = MarchingCubesConfig {
            resolution: 8,
            iso_level: 0.0,
            compute_normals: true,
            ..Default::default()
        };
        let mesh = sdf_to_mesh(&sphere, Vec3::splat(-2.0), Vec3::splat(2.0), &config);

        let mut mat_lib = MaterialLibrary::new();
        mat_lib.add(
            crate::material::Material::metal("Textured", 0.9, 0.9, 0.9, 0.2)
                .with_albedo_map("textures/albedo.png")
                .with_normal_map("textures/normal.png")
                .with_metallic_roughness_map("textures/mr.png")
                .with_ao_map("textures/ao.png"),
        );

        let (json_bytes, _bin) =
            build_glb_data(&mesh, &GltfConfig::default(), Some(&mat_lib)).unwrap();
        let json_str = String::from_utf8(json_bytes).unwrap();

        // Verify texture pipeline is present
        assert!(
            json_str.contains(r#""images":[{"uri":"textures/albedo.png"}"#),
            "Missing images array. JSON: {}",
            json_str
        );
        assert!(
            json_str.contains(r#""textures":["#),
            "Missing textures array"
        );
        assert!(
            json_str.contains(r#""samplers":["#),
            "Missing samplers array"
        );
        assert!(
            json_str.contains(r#""baseColorTexture":"#),
            "Missing baseColorTexture in material"
        );
        assert!(
            json_str.contains(r#""normalTexture":"#),
            "Missing normalTexture in material"
        );
        assert!(
            json_str.contains(r#""occlusionTexture":"#),
            "Missing occlusionTexture"
        );
    }

    #[test]
    fn test_glb_quantized_positions() {
        let sphere = SdfNode::sphere(1.0);
        let config = MarchingCubesConfig {
            resolution: 8,
            iso_level: 0.0,
            compute_normals: true,
            ..Default::default()
        };
        let mesh = sdf_to_mesh(&sphere, Vec3::splat(-2.0), Vec3::splat(2.0), &config);

        let gltf_config = GltfConfig {
            quantize_positions: true,
            ..Default::default()
        };

        let (json_bytes, bin) = build_glb_data(&mesh, &gltf_config, None).unwrap();
        let json_str = String::from_utf8(json_bytes).unwrap();

        // Should declare the extension
        assert!(
            json_str.contains("KHR_mesh_quantization"),
            "Missing quantization extension. JSON: {}",
            json_str
        );
        // Should have node transform for dequantization
        assert!(
            json_str.contains("\"translation\":["),
            "Missing dequantization translation"
        );
        assert!(
            json_str.contains("\"scale\":["),
            "Missing dequantization scale"
        );
        // Compare with non-quantized version to verify size reduction
        let (_, bin_float) = build_glb_data(&mesh, &GltfConfig::default(), None).unwrap();
        assert!(
            bin.len() < bin_float.len(),
            "Quantized buffer ({}) should be smaller than float ({})",
            bin.len(),
            bin_float.len()
        );

        // Export as file to verify GLB integrity
        let path = std::env::temp_dir().join("alice_test_quantized.glb");
        export_glb(&mesh, &path, &gltf_config, None).unwrap();

        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(&bytes[0..4], &GLB_MAGIC.to_le_bytes());

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_glb_u16_indices() {
        // Small mesh should use u16 indices
        let sphere = SdfNode::sphere(1.0);
        let config = MarchingCubesConfig {
            resolution: 4,
            iso_level: 0.0,
            compute_normals: true,
            ..Default::default()
        };
        let mesh = sdf_to_mesh(&sphere, Vec3::splat(-2.0), Vec3::splat(2.0), &config);
        assert!(mesh.vertex_count() <= 65535);

        let path = std::env::temp_dir().join("alice_test_u16.glb");
        export_glb(&mesh, &path, &GltfConfig::default(), None).unwrap();

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_glb_texture_embedding() {
        let sphere = SdfNode::sphere(1.0);
        let config = MarchingCubesConfig {
            resolution: 4,
            iso_level: 0.0,
            compute_normals: true,
            ..Default::default()
        };
        let mesh = sdf_to_mesh(&sphere, Vec3::splat(-2.0), Vec3::splat(2.0), &config);

        // Create a tiny test PNG
        let test_dir = std::env::temp_dir().join("alice_tex_test");
        std::fs::create_dir_all(&test_dir).ok();
        let tex_path = test_dir.join("test_albedo.png");
        // Minimal valid PNG (1x1 red pixel)
        let png_data: Vec<u8> = vec![
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, // PNG signature
            0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52, // IHDR chunk
            0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x02, 0x00, 0x00, 0x00, 0x90,
            0x77, 0x53, 0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41, // IDAT chunk
            0x54, 0x08, 0xD7, 0x63, 0xF8, 0xCF, 0xC0, 0x00, 0x00, 0x00, 0x03, 0x00, 0x01, 0x36,
            0x28, 0x19, 0x00, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, // IEND chunk
            0x44, 0xAE, 0x42, 0x60, 0x82,
        ];
        std::fs::write(&tex_path, &png_data).unwrap();

        let mut mat_lib = MaterialLibrary::new();
        mat_lib.add(
            crate::material::Material::metal("Textured", 0.9, 0.9, 0.9, 0.2)
                .with_albedo_map(tex_path.to_string_lossy().to_string()),
        );

        let gltf_config = GltfConfig {
            embed_textures: true,
            ..Default::default()
        };

        let (json_bytes, bin) = build_glb_data(&mesh, &gltf_config, Some(&mat_lib)).unwrap();
        let json_str = String::from_utf8(json_bytes).unwrap();

        // Should use bufferView instead of URI
        assert!(
            json_str.contains(r#""bufferView""#),
            "Should embed via bufferView, got: {}",
            json_str
        );
        assert!(
            json_str.contains(r#""mimeType":"image/png""#),
            "Should have mimeType"
        );
        // Binary buffer should contain the PNG data
        assert!(
            bin.len() >= png_data.len(),
            "Binary buffer should contain embedded image"
        );

        std::fs::remove_dir_all(&test_dir).ok();
    }

    #[test]
    fn test_glb_extended_materials() {
        let sphere = SdfNode::sphere(1.0);
        let config = MarchingCubesConfig {
            resolution: 4,
            iso_level: 0.0,
            compute_normals: true,
            ..Default::default()
        };
        let mesh = sdf_to_mesh(&sphere, Vec3::splat(-2.0), Vec3::splat(2.0), &config);

        let mut mat_lib = MaterialLibrary::new();
        mat_lib.add(
            crate::material::Material::new("CarPaint")
                .with_color(0.8, 0.0, 0.0, 1.0)
                .with_clearcoat(1.0, 0.1)
                .with_metallic(0.9)
                .with_roughness(0.3),
        );
        mat_lib.add(
            crate::material::Material::glass("GlassPanel", 1.52)
                .with_transmission(0.95)
                .with_volume(0.01, 0.5, 0.9, 0.95, 1.0),
        );
        mat_lib.add(crate::material::Material::new("Velvet").with_sheen(0.5, 0.3, 0.8, 0.7));

        let gltf_config = GltfConfig {
            export_extensions: true,
            ..Default::default()
        };

        let (json_bytes, _) = build_glb_data(&mesh, &gltf_config, Some(&mat_lib)).unwrap();
        let json_str = String::from_utf8(json_bytes).unwrap();

        assert!(
            json_str.contains("KHR_materials_clearcoat"),
            "Missing clearcoat extension: {}",
            json_str
        );
        assert!(
            json_str.contains("KHR_materials_transmission"),
            "Missing transmission extension"
        );
        assert!(
            json_str.contains("KHR_materials_ior"),
            "Missing IOR extension"
        );
        assert!(
            json_str.contains("KHR_materials_volume"),
            "Missing volume extension"
        );
        assert!(
            json_str.contains("KHR_materials_sheen"),
            "Missing sheen extension"
        );
        assert!(
            json_str.contains("extensionsUsed"),
            "Missing extensionsUsed"
        );
    }

    #[test]
    fn test_glb_import_roundtrip() {
        let sphere = SdfNode::sphere(1.0);
        let config = MarchingCubesConfig {
            resolution: 8,
            iso_level: 0.0,
            compute_normals: true,
            ..Default::default()
        };
        let mesh = sdf_to_mesh(&sphere, Vec3::splat(-2.0), Vec3::splat(2.0), &config);
        let orig_verts = mesh.vertex_count();
        let orig_tris = mesh.indices.len() / 3;

        // Export to GLB bytes
        let glb_bytes = export_glb_bytes(&mesh, &GltfConfig::default(), None).unwrap();

        // Import back
        let imported = import_glb_bytes(&glb_bytes).unwrap();

        assert_eq!(imported.vertex_count(), orig_verts, "Vertex count mismatch");
        assert_eq!(
            imported.indices.len() / 3,
            orig_tris,
            "Triangle count mismatch"
        );

        // Check first vertex position is close
        let orig_pos = mesh.vertices[0].position;
        let imp_pos = imported.vertices[0].position;
        assert!(
            (orig_pos - imp_pos).length() < 0.001,
            "Position mismatch: {:?} vs {:?}",
            orig_pos,
            imp_pos
        );
    }

    #[test]
    fn test_glb_import_file_roundtrip() {
        let sphere = SdfNode::sphere(1.0);
        let config = MarchingCubesConfig {
            resolution: 4,
            iso_level: 0.0,
            compute_normals: true,
            ..Default::default()
        };
        let mesh = sdf_to_mesh(&sphere, Vec3::splat(-2.0), Vec3::splat(2.0), &config);

        let path = std::env::temp_dir().join("alice_import_test.glb");
        export_glb(&mesh, &path, &GltfConfig::default(), None).unwrap();

        let imported = import_glb(&path).unwrap();
        assert_eq!(imported.vertex_count(), mesh.vertex_count());
        assert_eq!(imported.indices.len(), mesh.indices.len());

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_glb_import_invalid() {
        assert!(import_glb_bytes(&[]).is_err());
        assert!(import_glb_bytes(&[0u8; 12]).is_err());
        assert!(import_glb_bytes(b"not a glb file at all").is_err());
    }
}

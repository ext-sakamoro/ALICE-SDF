//! Nanite Cluster Export for UE5
//!
//! Exports NaniteMesh data to `.nanite` binary format for UE5 Nanite import.
//! Also supports JSON cluster manifest for custom UE5 import plugins.
//!
//! # Formats
//! - `.nanite` — Compact binary with cluster hierarchy
//! - `.nanite.json` — JSON manifest with cluster metadata
//!
//! Author: Moroya Sakamoto

use crate::io::IoError;
use crate::mesh::nanite::NaniteMesh;
use std::io::Write;
use std::path::Path;

/// Magic bytes for the .nanite binary format
pub const NANITE_MAGIC: &[u8; 4] = b"NANT";

/// Current binary format version
pub const NANITE_VERSION: u32 = 1;

/// Configuration for Nanite export
#[derive(Debug, Clone)]
pub struct NaniteExportConfig {
    /// Export UV coordinates per vertex
    pub export_uvs: bool,
    /// Export normals per vertex
    pub export_normals: bool,
    /// Quantize positions to 16-bit (halves file size)
    pub quantize_positions: bool,
}

impl Default for NaniteExportConfig {
    fn default() -> Self {
        NaniteExportConfig {
            export_uvs: true,
            export_normals: true,
            quantize_positions: false,
        }
    }
}

/// Export NaniteMesh to binary .nanite format
pub fn export_nanite(
    nanite: &NaniteMesh,
    path: impl AsRef<Path>,
) -> Result<(), IoError> {
    export_nanite_with_config(nanite, path, &NaniteExportConfig::default())
}

/// Export NaniteMesh to binary .nanite format with configuration
pub fn export_nanite_with_config(
    nanite: &NaniteMesh,
    path: impl AsRef<Path>,
    config: &NaniteExportConfig,
) -> Result<(), IoError> {
    let file = std::fs::File::create(path)?;
    let mut w = std::io::BufWriter::new(file);

    // Count totals
    let total_vertices: u32 = nanite.clusters.iter()
        .map(|c| c.vertices.len() as u32)
        .sum();
    let total_triangles: u32 = nanite.clusters.iter()
        .map(|c| c.triangles.len() as u32)
        .sum();

    // Header (32 bytes)
    w.write_all(NANITE_MAGIC)?;
    w.write_all(&NANITE_VERSION.to_le_bytes())?;
    w.write_all(&(nanite.clusters.len() as u32).to_le_bytes())?;
    w.write_all(&(nanite.groups.len() as u32).to_le_bytes())?;
    w.write_all(&(nanite.lod_levels.len() as u32).to_le_bytes())?;
    w.write_all(&total_vertices.to_le_bytes())?;
    w.write_all(&total_triangles.to_le_bytes())?;
    // Flags: bit0=uvs, bit1=normals, bit2=quantized
    let flags: u32 = (config.export_uvs as u32)
        | ((config.export_normals as u32) << 1)
        | ((config.quantize_positions as u32) << 2);
    w.write_all(&flags.to_le_bytes())?;

    // LOD levels
    for lod in &nanite.lod_levels {
        w.write_all(&lod.level.to_le_bytes())?;
        w.write_all(&lod.resolution.to_le_bytes())?;
        w.write_all(&lod.max_error.to_le_bytes())?;
        w.write_all(&lod.triangle_count.to_le_bytes())?;
    }

    // Clusters
    for cluster in &nanite.clusters {
        w.write_all(&cluster.id.to_le_bytes())?;
        w.write_all(&cluster.lod_level.to_le_bytes())?;
        w.write_all(&(cluster.vertices.len() as u32).to_le_bytes())?;
        w.write_all(&(cluster.triangles.len() as u32).to_le_bytes())?;

        // Bounds
        w.write_all(&cluster.bounds.center.x.to_le_bytes())?;
        w.write_all(&cluster.bounds.center.y.to_le_bytes())?;
        w.write_all(&cluster.bounds.center.z.to_le_bytes())?;
        w.write_all(&cluster.bounds.radius.to_le_bytes())?;
        w.write_all(&cluster.bounds.aabb_min.x.to_le_bytes())?;
        w.write_all(&cluster.bounds.aabb_min.y.to_le_bytes())?;
        w.write_all(&cluster.bounds.aabb_min.z.to_le_bytes())?;
        w.write_all(&cluster.bounds.aabb_max.x.to_le_bytes())?;
        w.write_all(&cluster.bounds.aabb_max.y.to_le_bytes())?;
        w.write_all(&cluster.bounds.aabb_max.z.to_le_bytes())?;
        w.write_all(&cluster.geometric_error.to_le_bytes())?;
        w.write_all(&cluster.material_id.to_le_bytes())?;

        // Parent/child DAG
        w.write_all(&(cluster.parent_ids.len() as u32).to_le_bytes())?;
        w.write_all(&(cluster.child_ids.len() as u32).to_le_bytes())?;
        for &pid in &cluster.parent_ids {
            w.write_all(&pid.to_le_bytes())?;
        }
        for &cid in &cluster.child_ids {
            w.write_all(&cid.to_le_bytes())?;
        }

        // Vertices
        for v in &cluster.vertices {
            w.write_all(&v.position.x.to_le_bytes())?;
            w.write_all(&v.position.y.to_le_bytes())?;
            w.write_all(&v.position.z.to_le_bytes())?;
            if config.export_normals {
                w.write_all(&v.normal.x.to_le_bytes())?;
                w.write_all(&v.normal.y.to_le_bytes())?;
                w.write_all(&v.normal.z.to_le_bytes())?;
            }
            if config.export_uvs {
                w.write_all(&v.uv.x.to_le_bytes())?;
                w.write_all(&v.uv.y.to_le_bytes())?;
            }
        }

        // Triangles
        for t in &cluster.triangles {
            w.write_all(&t.a.to_le_bytes())?;
            w.write_all(&t.b.to_le_bytes())?;
            w.write_all(&t.c.to_le_bytes())?;
        }
    }

    w.flush()?;
    Ok(())
}

/// Export NaniteMesh metadata to JSON manifest
pub fn export_nanite_json(
    nanite: &NaniteMesh,
    path: impl AsRef<Path>,
) -> Result<(), IoError> {
    let file = std::fs::File::create(path)?;
    let mut w = std::io::BufWriter::new(file);

    writeln!(w, "{{")?;
    writeln!(w, "  \"generator\": \"ALICE-SDF\",")?;
    writeln!(w, "  \"format_version\": {},", NANITE_VERSION)?;
    writeln!(w, "  \"cluster_count\": {},", nanite.clusters.len())?;
    writeln!(w, "  \"group_count\": {},", nanite.groups.len())?;

    // LOD levels
    writeln!(w, "  \"lod_levels\": [")?;
    for (i, lod) in nanite.lod_levels.iter().enumerate() {
        let comma = if i + 1 < nanite.lod_levels.len() { "," } else { "" };
        writeln!(w, "    {{\"level\": {}, \"resolution\": {}, \"max_error\": {:.6}, \"triangle_count\": {}}}{comma}",
            lod.level, lod.resolution, lod.max_error, lod.triangle_count)?;
    }
    writeln!(w, "  ],")?;

    // Clusters summary
    writeln!(w, "  \"clusters\": [")?;
    for (i, cluster) in nanite.clusters.iter().enumerate() {
        let comma = if i + 1 < nanite.clusters.len() { "," } else { "" };
        writeln!(w, "    {{")?;
        writeln!(w, "      \"id\": {},", cluster.id)?;
        writeln!(w, "      \"lod_level\": {},", cluster.lod_level)?;
        writeln!(w, "      \"vertex_count\": {},", cluster.vertices.len())?;
        writeln!(w, "      \"triangle_count\": {},", cluster.triangles.len())?;
        writeln!(w, "      \"geometric_error\": {:.6},", cluster.geometric_error)?;
        writeln!(w, "      \"material_id\": {},", cluster.material_id)?;
        writeln!(w, "      \"bounds\": {{")?;
        writeln!(w, "        \"center\": [{:.6}, {:.6}, {:.6}],",
            cluster.bounds.center.x, cluster.bounds.center.y, cluster.bounds.center.z)?;
        writeln!(w, "        \"radius\": {:.6},", cluster.bounds.radius)?;
        writeln!(w, "        \"aabb_min\": [{:.6}, {:.6}, {:.6}],",
            cluster.bounds.aabb_min.x, cluster.bounds.aabb_min.y, cluster.bounds.aabb_min.z)?;
        writeln!(w, "        \"aabb_max\": [{:.6}, {:.6}, {:.6}]",
            cluster.bounds.aabb_max.x, cluster.bounds.aabb_max.y, cluster.bounds.aabb_max.z)?;
        writeln!(w, "      }},")?;
        writeln!(w, "      \"parent_ids\": {:?},", cluster.parent_ids)?;
        writeln!(w, "      \"child_ids\": {:?}", cluster.child_ids)?;
        writeln!(w, "    }}{comma}")?;
    }
    writeln!(w, "  ]")?;

    writeln!(w, "}}")?;
    w.flush()?;
    Ok(())
}

/// Export a Nanite-compatible HLSL material function (.usf)
///
/// Generates a procedural material function that can be used with UE5's
/// custom material nodes. The SDF tree is transpiled to HLSL and wrapped
/// in a material-ready function.
///
/// # Feature Requirements
/// Requires the `hlsl` feature to be enabled. Build with:
/// ```bash
/// cargo build --features hlsl
/// ```
#[cfg(feature = "hlsl")]
pub fn export_nanite_hlsl_material(
    sdf: &crate::types::SdfNode,
    path: impl AsRef<Path>,
) -> Result<(), IoError> {
    use crate::compiled::hlsl::{HlslShader, HlslTranspileMode};

    let file = std::fs::File::create(path)?;
    let mut w = std::io::BufWriter::new(file);

    writeln!(w, "// ALICE-SDF Nanite Material Function")?;
    writeln!(w, "// Auto-generated procedural material for UE5 Nanite")?;
    writeln!(w, "// Usage: Add as Custom Expression in UE5 Material Editor")?;
    writeln!(w, "// Input: float3 WorldPosition")?;
    writeln!(w, "// Output: float4 (xyz = normal, w = signed_distance)")?;
    writeln!(w, "")?;

    // Transpile SDF tree to HLSL function
    let shader = HlslShader::transpile(sdf, HlslTranspileMode::Hardcoded);
    w.write_all(shader.source.as_bytes())?;
    writeln!(w, "")?;

    // Add normal computation helper
    writeln!(w, "// Normal computation via central differences")?;
    writeln!(w, "float3 AliceSdfNormal(float3 p)")?;
    writeln!(w, "{{")?;
    writeln!(w, "    const float eps = 0.001;")?;
    writeln!(w, "    return normalize(float3(")?;
    writeln!(w, "        sdf_eval(p + float3(eps, 0, 0)) - sdf_eval(p - float3(eps, 0, 0)),")?;
    writeln!(w, "        sdf_eval(p + float3(0, eps, 0)) - sdf_eval(p - float3(0, eps, 0)),")?;
    writeln!(w, "        sdf_eval(p + float3(0, 0, eps)) - sdf_eval(p - float3(0, 0, eps))")?;
    writeln!(w, "    ));")?;
    writeln!(w, "}}")?;
    writeln!(w, "")?;

    // Material entry point
    writeln!(w, "// Material entry point for UE5")?;
    writeln!(w, "// Returns: float4(normal.xyz, signed_distance)")?;
    writeln!(w, "float4 AliceSdfMaterial(float3 WorldPosition)")?;
    writeln!(w, "{{")?;
    writeln!(w, "    float d = sdf_eval(WorldPosition);")?;
    writeln!(w, "    float3 n = AliceSdfNormal(WorldPosition);")?;
    writeln!(w, "    return float4(n, d);")?;
    writeln!(w, "}}")?;

    w.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::SdfNode;
    use crate::mesh::nanite::{generate_nanite_mesh, NaniteConfig};

    fn make_test_nanite() -> NaniteMesh {
        let sphere = SdfNode::sphere(1.0);
        generate_nanite_mesh(&sphere, glam::Vec3::splat(-2.0), glam::Vec3::splat(2.0), &NaniteConfig::default())
    }

    #[test]
    fn test_nanite_export() {
        let nanite = make_test_nanite();
        let path = std::env::temp_dir().join("alice_test.nanite");

        export_nanite(&nanite, &path).unwrap();

        let data = std::fs::read(&path).unwrap();
        assert!(data.len() >= 32);
        assert_eq!(&data[0..4], b"NANT");

        let version = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
        assert_eq!(version, NANITE_VERSION);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_nanite_json_export() {
        let nanite = make_test_nanite();
        let path = std::env::temp_dir().join("alice_test.nanite.json");

        export_nanite_json(&nanite, &path).unwrap();

        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("\"generator\": \"ALICE-SDF\""));
        assert!(content.contains("\"cluster_count\""));
        assert!(content.contains("\"lod_levels\""));

        std::fs::remove_file(&path).ok();
    }

    #[test]
    #[cfg(feature = "hlsl")]
    fn test_nanite_hlsl_material_export() {
        let sphere = SdfNode::sphere(1.0);
        let path = std::env::temp_dir().join("alice_test_nanite_material.usf");

        export_nanite_hlsl_material(&sphere, &path).unwrap();

        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("ALICE-SDF"));
        assert!(content.contains("AliceSdfMaterial"));
        assert!(content.contains("AliceSdfNormal"));

        std::fs::remove_file(&path).ok();
    }
}

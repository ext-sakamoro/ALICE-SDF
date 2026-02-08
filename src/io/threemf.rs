//! 3MF (3D Manufacturing Format) export for ALICE-SDF meshes
//!
//! 3MF is a ZIP-based format containing XML files, designed for
//! additive manufacturing (3D printing). Uses STORE (no compression)
//! for simplicity and maximum compatibility.
//!
//! Author: Moroya Sakamoto

use crate::io::IoError;
use crate::mesh::Mesh;
use std::io::{Seek, SeekFrom, Write};
use std::path::Path;

/// Export mesh to 3MF format
pub fn export_3mf(mesh: &Mesh, path: impl AsRef<Path>) -> Result<(), IoError> {
    let file = std::fs::File::create(path)?;
    let w = std::io::BufWriter::new(file);
    let mut zip = ZipWriter::new(w);

    // [Content_Types].xml
    let content_types = r#"<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>
</Types>"#;
    zip.add_file("[Content_Types].xml", content_types.as_bytes())?;

    // _rels/.rels
    let rels = r#"<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>
</Relationships>"#;
    zip.add_file("_rels/.rels", rels.as_bytes())?;

    // 3D/3dmodel.model
    let mut model = String::with_capacity(mesh.vertices.len() * 40 + mesh.indices.len() * 15);
    model.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    model.push_str("<model unit=\"millimeter\" xmlns=\"http://schemas.microsoft.com/3dmanufacturing/core/2015/02\">\n");
    model.push_str("  <resources>\n");
    model.push_str("    <object id=\"1\" type=\"model\">\n");
    model.push_str("      <mesh>\n");
    model.push_str("        <vertices>\n");

    for v in &mesh.vertices {
        model.push_str(&format!(
            "          <vertex x=\"{}\" y=\"{}\" z=\"{}\"/>\n",
            v.position.x, v.position.y, v.position.z
        ));
    }

    model.push_str("        </vertices>\n");
    model.push_str("        <triangles>\n");

    let tri_count = mesh.indices.len() / 3;
    for i in 0..tri_count {
        model.push_str(&format!(
            "          <triangle v1=\"{}\" v2=\"{}\" v3=\"{}\"/>\n",
            mesh.indices[i * 3],
            mesh.indices[i * 3 + 1],
            mesh.indices[i * 3 + 2]
        ));
    }

    model.push_str("        </triangles>\n");
    model.push_str("      </mesh>\n");
    model.push_str("    </object>\n");
    model.push_str("  </resources>\n");
    model.push_str("  <build>\n");
    model.push_str("    <item objectid=\"1\"/>\n");
    model.push_str("  </build>\n");
    model.push_str("</model>\n");

    zip.add_file("3D/3dmodel.model", model.as_bytes())?;
    zip.finish()?;

    Ok(())
}

/// Minimal ZIP writer (STORE method, no compression)
struct ZipWriter<W: Write + Seek> {
    writer: W,
    entries: Vec<ZipEntry>,
}

struct ZipEntry {
    name: Vec<u8>,
    offset: u32,
    size: u32,
    crc32: u32,
}

impl<W: Write + Seek> ZipWriter<W> {
    fn new(writer: W) -> Self {
        ZipWriter {
            writer,
            entries: Vec::new(),
        }
    }

    fn add_file(&mut self, name: &str, data: &[u8]) -> Result<(), IoError> {
        let offset = self.writer.stream_position()
            .map_err(|e| IoError::Io(e))? as u32;
        let crc = crc32fast::hash(data);
        let size = data.len() as u32;
        let name_bytes = name.as_bytes();

        // Local file header
        self.writer.write_all(&0x04034b50u32.to_le_bytes())?; // signature
        self.writer.write_all(&20u16.to_le_bytes())?; // version needed
        self.writer.write_all(&0u16.to_le_bytes())?; // flags
        self.writer.write_all(&0u16.to_le_bytes())?; // compression (STORE)
        self.writer.write_all(&0u16.to_le_bytes())?; // mod time
        self.writer.write_all(&0u16.to_le_bytes())?; // mod date
        self.writer.write_all(&crc.to_le_bytes())?; // crc32
        self.writer.write_all(&size.to_le_bytes())?; // compressed size
        self.writer.write_all(&size.to_le_bytes())?; // uncompressed size
        self.writer.write_all(&(name_bytes.len() as u16).to_le_bytes())?; // name len
        self.writer.write_all(&0u16.to_le_bytes())?; // extra len
        self.writer.write_all(name_bytes)?;
        self.writer.write_all(data)?;

        self.entries.push(ZipEntry {
            name: name_bytes.to_vec(),
            offset,
            size,
            crc32: crc,
        });

        Ok(())
    }

    fn finish(mut self) -> Result<(), IoError> {
        let cd_offset = self.writer.stream_position()
            .map_err(|e| IoError::Io(e))? as u32;

        // Central directory entries
        for entry in &self.entries {
            self.writer.write_all(&0x02014b50u32.to_le_bytes())?; // signature
            self.writer.write_all(&20u16.to_le_bytes())?; // version made by
            self.writer.write_all(&20u16.to_le_bytes())?; // version needed
            self.writer.write_all(&0u16.to_le_bytes())?; // flags
            self.writer.write_all(&0u16.to_le_bytes())?; // compression
            self.writer.write_all(&0u16.to_le_bytes())?; // mod time
            self.writer.write_all(&0u16.to_le_bytes())?; // mod date
            self.writer.write_all(&entry.crc32.to_le_bytes())?;
            self.writer.write_all(&entry.size.to_le_bytes())?; // compressed
            self.writer.write_all(&entry.size.to_le_bytes())?; // uncompressed
            self.writer.write_all(&(entry.name.len() as u16).to_le_bytes())?;
            self.writer.write_all(&0u16.to_le_bytes())?; // extra len
            self.writer.write_all(&0u16.to_le_bytes())?; // comment len
            self.writer.write_all(&0u16.to_le_bytes())?; // disk number
            self.writer.write_all(&0u16.to_le_bytes())?; // internal attr
            self.writer.write_all(&0u32.to_le_bytes())?; // external attr
            self.writer.write_all(&entry.offset.to_le_bytes())?;
            self.writer.write_all(&entry.name)?;
        }

        let cd_end = self.writer.stream_position()
            .map_err(|e| IoError::Io(e))? as u32;
        let cd_size = cd_end - cd_offset;
        let entry_count = self.entries.len() as u16;

        // End of central directory
        self.writer.write_all(&0x06054b50u32.to_le_bytes())?; // signature
        self.writer.write_all(&0u16.to_le_bytes())?; // disk number
        self.writer.write_all(&0u16.to_le_bytes())?; // start disk
        self.writer.write_all(&entry_count.to_le_bytes())?;
        self.writer.write_all(&entry_count.to_le_bytes())?;
        self.writer.write_all(&cd_size.to_le_bytes())?;
        self.writer.write_all(&cd_offset.to_le_bytes())?;
        self.writer.write_all(&0u16.to_le_bytes())?; // comment len

        self.writer.flush()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::{sdf_to_mesh, MarchingCubesConfig};
    use crate::types::SdfNode;
    use glam::Vec3;

    #[test]
    fn test_3mf_export() {
        let sphere = SdfNode::sphere(1.0);
        let config = MarchingCubesConfig {
            resolution: 8,
            iso_level: 0.0,
            compute_normals: true,
            ..Default::default()
        };
        let mesh = sdf_to_mesh(&sphere, Vec3::splat(-2.0), Vec3::splat(2.0), &config);
        let path = std::env::temp_dir().join("alice_test.3mf");
        export_3mf(&mesh, &path).unwrap();
        let meta = std::fs::metadata(&path).unwrap();
        assert!(meta.len() > 100);
        std::fs::remove_file(&path).ok();
    }
}

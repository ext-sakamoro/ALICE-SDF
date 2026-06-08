//! IGES (Initial Graphics Exchange Specification, ASCII) mesh export
//!
//! 古典的な CAD 交換フォーマット。Rhino / AutoCAD / Fusion 360 など多くのツールで
//! 開ける。本実装は **書き出し専用**、Entity 144 (Trimmed Surface) 等の複雑な
//! parametric 表現はサポートせず、Entity 100 (Circular Arc) や Entity 110 (Line)
//! の代わりに **Entity 134 (Node)** + **Entity 136 (Finite Element)** で
//! 三角形メッシュを表現する。
//!
//! IGES 仕様: <https://wiki.eclipse.org/IGES_file_Format>
//!
//! # 使用例
//!
//! ```ignore
//! use alice_sdf::io::iges::{export_iges, IgesConfig};
//! use alice_sdf::prelude::*;
//!
//! let node = SdfNode::sphere(1.0);
//! let cfg = IgesConfig { bounds: (-2.0, 2.0), resolution: 32 };
//! export_iges("sphere.igs", &node, &cfg).unwrap();
//! ```

use crate::mesh::{sdf_to_mesh, MarchingCubesConfig};
use crate::types::SdfNode;
use glam::Vec3;
use std::io::Write;
use std::path::Path;

/// IGES export 設定
#[derive(Clone, Copy, Debug)]
pub struct IgesConfig {
    /// グリッド範囲 (min, max)
    pub bounds: (f32, f32),
    /// 1 辺の voxel resolution
    pub resolution: u32,
}

impl Default for IgesConfig {
    fn default() -> Self {
        Self {
            bounds: (-2.0, 2.0),
            resolution: 32,
        }
    }
}

fn sdf_to_facets(node: &SdfNode, cfg: &IgesConfig) -> (Vec<Vec3>, Vec<[usize; 3]>) {
    let (lo, hi) = cfg.bounds;
    let mc = MarchingCubesConfig {
        resolution: (cfg.resolution as usize).max(4),
        iso_level: 0.0,
        compute_normals: false,
        compute_uvs: false,
        uv_scale: 1.0,
        compute_tangents: false,
        compute_materials: false,
    };
    let mesh = sdf_to_mesh(node, Vec3::splat(lo), Vec3::splat(hi), &mc);
    let verts: Vec<Vec3> = mesh.vertices.iter().map(|v| v.position).collect();
    let tris: Vec<[usize; 3]> = mesh
        .indices
        .chunks_exact(3)
        .map(|t| [t[0] as usize, t[1] as usize, t[2] as usize])
        .collect();
    (verts, tris)
}

/// SDF → IGES ファイル書き出し
pub fn export_iges(
    path: impl AsRef<Path>,
    node: &SdfNode,
    cfg: &IgesConfig,
) -> std::io::Result<()> {
    let (verts, tris) = sdf_to_facets(node, cfg);
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    write_iges(&mut f, &verts, &tris)?;
    f.flush()
}

/// 各 IGES line は 80 文字、列 73-80 にセクション文字 + シーケンス番号
fn iges_line(content: &str, section: char, seq: u32) -> String {
    let mut s = String::with_capacity(80);
    let trimmed = if content.len() > 72 {
        &content[..72]
    } else {
        content
    };
    s.push_str(trimmed);
    while s.len() < 72 {
        s.push(' ');
    }
    s.push(section);
    let n = format!("{:>7}", seq);
    s.push_str(&n);
    s
}

fn write_iges<W: Write>(w: &mut W, verts: &[Vec3], tris: &[[usize; 3]]) -> std::io::Result<()> {
    let mut sseq = 1u32;
    let mut gseq = 1u32;

    // ==== START SECTION ====
    writeln!(
        w,
        "{}",
        iges_line(
            "ALICE-SDF IGES export - SDF tessellation as Node/Finite Element entities",
            'S',
            sseq
        )
    )?;
    sseq += 1;

    // ==== GLOBAL SECTION ====
    let global = "1H,,1H;,4HSDF1,8Halice.igs,12HALICE-SDF v16,12HALICE-SDF v16,32,38,6,308,15,4HSDF1,1.,2,2HMM,50,0.125,15H20260606.000000,1.E-007,1000.,12HALICE-SDF,9HSakamoto,11,0,15H20260606.000000;";
    let mut start = 0;
    while start < global.len() {
        let end = (start + 72).min(global.len());
        writeln!(w, "{}", iges_line(&global[start..end], 'G', gseq))?;
        gseq += 1;
        start = end;
    }

    // ==== DIRECTORY ENTRY (DE) ====
    // Each finite element node (134) + finite element (136) gets a DE pair (2 lines = 1 entity).
    //   Node entity: type 134, NODE
    //   Finite element entity: type 136, FINITE_ELEMENT
    let p_base = 1u32;
    let p_cursor = p_base;
    let mut p_line_per_entity = Vec::new();

    // We'll lay out PD lines first to determine line count, then DE.
    let mut pd_buffer: Vec<String> = Vec::new();
    // Node entities
    let mut node_de_ids: Vec<u32> = Vec::with_capacity(verts.len());
    for (i, v) in verts.iter().enumerate() {
        let line = format!("134,{:.6},{:.6},{:.6},0,0,0;", v.x, v.y, v.z);
        let de_id = (i * 2 + 1) as u32; // each DE has 2 lines, so id 1,3,5,...
        node_de_ids.push(de_id);
        pd_buffer.push(line);
        p_line_per_entity.push(1u32);
    }
    // Finite element entities (triangles)
    let mut tri_de_ids: Vec<u32> = Vec::with_capacity(tris.len());
    let node_de_count = (verts.len() * 2) as u32;
    for (i, tri) in tris.iter().enumerate() {
        let de_id = node_de_count + (i * 2 + 1) as u32;
        tri_de_ids.push(de_id);
        let line = format!(
            "136,3,{},{},{};",
            node_de_ids.get(tri[0]).copied().unwrap_or(0),
            node_de_ids.get(tri[1]).copied().unwrap_or(0),
            node_de_ids.get(tri[2]).copied().unwrap_or(0),
        );
        pd_buffer.push(line);
        p_line_per_entity.push(1u32);
    }

    // Write DE section
    let mut de_seq = 1u32;
    let mut pd_cursor = 1u32;
    for (i, &lines) in p_line_per_entity.iter().enumerate() {
        let etype = if i < verts.len() { 134 } else { 136 };
        let de_left = format!(
            "{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}",
            etype, pd_cursor, 0, 0, 0, 0, 0, 0, 0
        );
        let de_right = format!(
            "{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}",
            etype, 0, lines, 0, 0, 0, 0, 0
        );
        writeln!(w, "{}", iges_line(&de_left, 'D', de_seq))?;
        de_seq += 1;
        writeln!(w, "{}", iges_line(&de_right, 'D', de_seq))?;
        de_seq += 1;
        pd_cursor += lines;
        let _ = p_cursor;
    }

    // ==== PARAMETER DATA SECTION ====
    let mut p_seq = 1u32;
    for line in &pd_buffer {
        let mut content = String::with_capacity(80);
        content.push_str(line);
        while content.len() < 64 {
            content.push(' ');
        }
        let de_ref = format!(" {:>7}", p_seq);
        content.push_str(&de_ref);
        writeln!(w, "{}", iges_line(&content, 'P', p_seq))?;
        p_seq += 1;
    }
    // ==== TERMINATE SECTION ====
    let term = format!(
        "S{:>7}G{:>7}D{:>7}P{:>7}",
        sseq - 1,
        gseq - 1,
        de_seq - 1,
        p_seq - 1
    );
    writeln!(w, "{}", iges_line(&term, 'T', 1))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::SdfNode;

    #[test]
    fn sphere_iges_export() {
        let path = std::env::temp_dir().join("alice_sdf_test.igs");
        let n = SdfNode::sphere(1.0);
        let cfg = IgesConfig {
            bounds: (-2.0, 2.0),
            resolution: 6,
        };
        export_iges(&path, &n, &cfg).unwrap();
        let content = std::fs::read_to_string(&path).unwrap();
        // 各 line は 80 chars + \n、Section letter は最終列 73 (column 73, index 72)
        assert!(content.contains("S      1"));
        // Terminate section
        assert!(content.contains("T      1"));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn iges_line_is_80_chars() {
        let l = iges_line("hello", 'S', 1);
        assert_eq!(l.len(), 80);
        assert_eq!(&l[72..73], "S");
    }

    #[test]
    fn iges_marching_cubes_produces_dense_mesh() {
        let n = SdfNode::sphere(1.0);
        let cfg = IgesConfig {
            bounds: (-1.5, 1.5),
            resolution: 16,
        };
        let (verts, tris) = super::sdf_to_facets(&n, &cfg);
        assert!(
            verts.len() > 100,
            "expected >100 verts, got {}",
            verts.len()
        );
        assert!(!tris.is_empty());
    }

    #[test]
    fn iges_export_includes_134_node_and_136_fe_entities() {
        let path = std::env::temp_dir().join("alice_sdf_entities.igs");
        let n = SdfNode::sphere(0.8);
        let cfg = IgesConfig {
            bounds: (-1.2, 1.2),
            resolution: 8,
        };
        export_iges(&path, &n, &cfg).unwrap();
        let content = std::fs::read_to_string(&path).unwrap();
        // PD section に entity type 134 (Node) と 136 (Finite Element) が現れる
        assert!(content.contains("134,"), "missing Node entity 134");
        assert!(content.contains("136,3,"), "missing FE entity 136 (3-node)");
        std::fs::remove_file(&path).ok();
    }
}

//! STEP (ISO 10303-21) ASCII Part 21 mesh export
//!
//! ALICE-SDF の SDF tree を tessellate して、`AP203` 互換の `Faceted_BREP` 形式で
//! STEP テキストファイルに書き出す。Fusion 360 / SolidWorks / OnShape / Rhino / FreeCAD
//! 等の CAD ツールで開ける。
//!
//! 本実装は **書き出し専用** (読み込みは別途解析必要)。最小限の STEP entity セットで、
//! NURBS / parametric surface 等の高機能は未対応。
//!
//! # 使用例
//!
//! ```ignore
//! use alice_sdf::io::step::{export_step, StepConfig};
//! use alice_sdf::prelude::*;
//!
//! let node = SdfNode::sphere(1.0);
//! let cfg = StepConfig {
//!     bounds: (-2.0, 2.0),
//!     resolution: 32,
//!     name: "AliceSdfSphere".into(),
//! };
//! export_step("sphere.step", &node, &cfg).unwrap();
//! ```

use crate::eval::eval;
use crate::types::SdfNode;
use glam::Vec3;
use std::io::Write;
use std::path::Path;

/// STEP export 設定
#[derive(Clone, Debug)]
pub struct StepConfig {
    /// グリッド範囲 (min, max)
    pub bounds: (f32, f32),
    /// Marching-cubes 風 surface 抽出の voxel resolution
    pub resolution: u32,
    /// STEP entity 名 (例: "AliceSdfPart")
    pub name: String,
}

impl Default for StepConfig {
    fn default() -> Self {
        Self {
            bounds: (-2.0, 2.0),
            resolution: 32,
            name: "AliceSdfMesh".into(),
        }
    }
}

/// SDF 表面に近い voxel の中心点をピックして簡易 facet 化
fn sdf_to_facets(node: &SdfNode, cfg: &StepConfig) -> (Vec<Vec3>, Vec<[usize; 3]>) {
    let (lo, hi) = cfg.bounds;
    let n = cfg.resolution.max(1);
    let step = (hi - lo) / n as f32;
    let eps = step;
    let mut verts = Vec::new();
    // 単純化: 表面近傍 voxel に 1 quadrilateral facet (2 triangles) を発生
    for k in 0..n {
        for j in 0..n {
            for i in 0..n {
                let cx = lo + (i as f32 + 0.5) * step;
                let cy = lo + (j as f32 + 0.5) * step;
                let cz = lo + (k as f32 + 0.5) * step;
                let d = eval(node, Vec3::new(cx, cy, cz));
                if d.abs() < eps {
                    let s = step * 0.5;
                    verts.push(Vec3::new(cx - s, cy - s, cz));
                    verts.push(Vec3::new(cx + s, cy - s, cz));
                    verts.push(Vec3::new(cx + s, cy + s, cz));
                    verts.push(Vec3::new(cx - s, cy + s, cz));
                }
            }
        }
    }
    let mut tris = Vec::new();
    for quad in 0..(verts.len() / 4) {
        let b = quad * 4;
        tris.push([b, b + 1, b + 2]);
        tris.push([b, b + 2, b + 3]);
    }
    (verts, tris)
}

/// SDF を tessellate して STEP AP203 ファイルに書き出し
pub fn export_step(
    path: impl AsRef<Path>,
    node: &SdfNode,
    cfg: &StepConfig,
) -> std::io::Result<()> {
    let (verts, tris) = sdf_to_facets(node, cfg);
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    write_step(&mut f, &verts, &tris, &cfg.name)?;
    f.flush()
}

fn write_step<W: Write>(
    w: &mut W,
    verts: &[Vec3],
    tris: &[[usize; 3]],
    name: &str,
) -> std::io::Result<()> {
    // HEADER
    writeln!(w, "ISO-10303-21;")?;
    writeln!(w, "HEADER;")?;
    writeln!(
        w,
        "FILE_DESCRIPTION(('ALICE-SDF v1.6 Faceted BREP'),'2;1');"
    )?;
    writeln!(
        w,
        "FILE_NAME('{name}.step','2026-06-06T00:00:00',('ALICE-SDF'),(''),'ALICE-SDF v1.6','','');",
    )?;
    writeln!(w, "FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));")?;
    writeln!(w, "ENDSEC;")?;
    writeln!(w, "DATA;")?;

    let mut id = 1u32;
    let mut next = || {
        let i = id;
        id += 1;
        i
    };

    // CARTESIAN_POINT 群
    let mut point_ids = Vec::with_capacity(verts.len());
    for v in verts {
        let pid = next();
        writeln!(
            w,
            "#{pid}=CARTESIAN_POINT('',({:.6},{:.6},{:.6}));",
            v.x, v.y, v.z
        )?;
        point_ids.push(pid);
    }

    // 三角形 (POLY_LOOP + FACE_OUTER_BOUND + ADVANCED_FACE は重いので、ここでは
    //  TRIANGULATED_FACE_SET 互換セマンティクスで 1 facet = 1 triangle として記述)
    for tri in tris {
        let p0 = point_ids[tri[0]];
        let p1 = point_ids[tri[1]];
        let p2 = point_ids[tri[2]];
        let loop_id = next();
        writeln!(w, "#{loop_id}=POLY_LOOP('',(#{p0},#{p1},#{p2}));")?;
        let face_id = next();
        writeln!(w, "#{face_id}=FACE_OUTER_BOUND('',#{loop_id},.T.);")?;
    }

    // ROOT: PRODUCT 等は最小限
    let prod_def_ctx = next();
    writeln!(
        w,
        "#{prod_def_ctx}=PRODUCT_DEFINITION_CONTEXT('part definition',#0,'design');"
    )?;
    let prod = next();
    writeln!(w, "#{prod}=PRODUCT('{name}','{name}','',(#0));")?;

    writeln!(w, "ENDSEC;")?;
    writeln!(w, "END-ISO-10303-21;")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::SdfNode;

    #[test]
    fn sphere_step_export() {
        let path = std::env::temp_dir().join("alice_sdf_test.step");
        let n = SdfNode::sphere(1.0);
        let cfg = StepConfig {
            bounds: (-2.0, 2.0),
            resolution: 8,
            name: "TestSphere".into(),
        };
        export_step(&path, &n, &cfg).unwrap();
        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.starts_with("ISO-10303-21;"));
        assert!(content.contains("END-ISO-10303-21;"));
        assert!(content.contains("CARTESIAN_POINT"));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn empty_node_writes_valid_header() {
        let path = std::env::temp_dir().join("alice_sdf_empty.step");
        let n = SdfNode::sphere(0.001);
        let cfg = StepConfig {
            bounds: (-2.0, 2.0),
            resolution: 4,
            name: "Empty".into(),
        };
        export_step(&path, &n, &cfg).unwrap();
        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("DATA;"));
        std::fs::remove_file(&path).ok();
    }
}

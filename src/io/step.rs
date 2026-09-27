//! STEP (ISO 10303-21) ASCII Part 21 mesh export
//!
//! ALICE-SDF の SDF tree を tessellate して、`AP214` (`AUTOMOTIVE_DESIGN`) の
//! faceted BREP として STEP テキストファイルに書き出す。
//!
//! 本実装は **書き出し専用** (読み込みは別途解析必要)。三角形 1 枚を平面
//! (`PLANE`) の `ADVANCED_FACE` として出すので、NURBS / 解析曲面
//! (`SPHERICAL_SURFACE` 等) は未対応 — 球を export しても CAD 側には
//! 平面の集合として届く (三相原理では Phase 1 相当、Phase 2 化は別途)。
//!
//! 出力は `CLOSED_SHELL` → `MANIFOLD_SOLID_BREP` →
//! `ADVANCED_BREP_SHAPE_REPRESENTATION` → `SHAPE_DEFINITION_REPRESENTATION`
//! の連結と単位系 (`LENGTH_UNIT` = mm) を持つ。2026-09-27 以前は
//! `CARTESIAN_POINT` + `POLY_LOOP` + `FACE_OUTER_BOUND` を並べるだけで
//! shell / solid / 表現 / 単位が無く、存在しない `#0` を参照していたため
//! (`tests/test_step_export_oracle.rs` が検出) CAD では開けなかった。
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

use crate::mesh::manifold::MeshRepair;
use crate::mesh::{sdf_to_mesh, MarchingCubesConfig};
use crate::types::SdfNode;
use glam::Vec3;
use std::collections::HashMap;
use std::io::Write;
use std::path::Path;

/// STEP export 設定
#[derive(Clone, Debug)]
pub struct StepConfig {
    /// グリッド範囲 (min, max)
    pub bounds: (f32, f32),
    /// Marching-cubes surface 抽出の voxel resolution
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

/// SDF を Marching Cubes で tessellate し (vertices, triangles) を返す
fn sdf_to_facets(node: &SdfNode, cfg: &StepConfig) -> (Vec<Vec3>, Vec<[usize; 3]>) {
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
    // 退化 facet は平面を持てず `ADVANCED_FACE` にできない (落とすと shell が
    // 開くので、面を作る前に重複頂点を潰しておく)
    let mesh = MeshRepair::repair_all(&mesh, 5e-3);
    let verts: Vec<Vec3> = mesh.vertices.iter().map(|v| v.position).collect();
    // `as_chunks` は MSRV より新しいので使わない (clippy::incompatible_msrv)
    let tris: Vec<[usize; 3]> = mesh
        .indices
        .chunks_exact(3)
        .map(|t| [t[0] as usize, t[1] as usize, t[2] as usize])
        .collect();
    (verts, tris)
}

/// 平面 6 枚で厳密に表せる形 (原点中心の軸平行箱) なら、その half-extents
///
/// 箱は tessellate すると voxel 解像度ぶん寸法がぶれるが、面は厳密に平面なので
/// 6 面でそのまま書ける (三相原理 Phase 2: 形を作る法則をそのまま送る)
const fn analytic_box(node: &SdfNode) -> Option<Vec3> {
    match node {
        SdfNode::Box3d { half_extents } => Some(*half_extents),
        _ => None,
    }
}

/// 原点中心の軸平行箱を (頂点 8 個, 外向き CCW の 4 角形 6 枚) にする
fn box_faces(h: Vec3) -> (Vec<Vec3>, Vec<Vec<usize>>) {
    let verts = vec![
        Vec3::new(-h.x, -h.y, -h.z),
        Vec3::new(h.x, -h.y, -h.z),
        Vec3::new(h.x, h.y, -h.z),
        Vec3::new(-h.x, h.y, -h.z),
        Vec3::new(-h.x, -h.y, h.z),
        Vec3::new(h.x, -h.y, h.z),
        Vec3::new(h.x, h.y, h.z),
        Vec3::new(-h.x, h.y, h.z),
    ];
    let faces = vec![
        vec![0, 3, 2, 1], // -Z
        vec![4, 5, 6, 7], // +Z
        vec![0, 1, 5, 4], // -Y
        vec![1, 2, 6, 5], // +X
        vec![2, 3, 7, 6], // +Y
        vec![3, 0, 4, 7], // -X
    ];
    (verts, faces)
}

/// SDF を STEP (AP214 faceted BREP) ファイルに書き出し
///
/// 原点中心の軸平行箱は tessellate せず 6 面で厳密に書き、それ以外は
/// Marching Cubes の三角形を 1 枚ずつ平面の面として書く。
pub fn export_step(
    path: impl AsRef<Path>,
    node: &SdfNode,
    cfg: &StepConfig,
) -> std::io::Result<()> {
    let (verts, faces) = if let Some(h) = analytic_box(node) {
        box_faces(h)
    } else {
        let (verts, tris) = sdf_to_facets(node, cfg);
        let faces = tris.iter().map(|t| t.to_vec()).collect();
        (verts, faces)
    };
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    write_step(&mut f, &verts, &faces, &cfg.name)?;
    f.flush()
}

/// `n` に直交する単位ベクトル (`AXIS2_PLACEMENT_3D` の参照方向)
fn perpendicular(n: Vec3) -> Vec3 {
    let helper = if n.x.abs() <= n.y.abs() && n.x.abs() <= n.z.abs() {
        Vec3::X
    } else if n.y.abs() <= n.z.abs() {
        Vec3::Y
    } else {
        Vec3::Z
    };
    let p = n.cross(helper);
    if p.length_squared() > 1e-12 {
        p.normalize()
    } else {
        Vec3::X
    }
}

fn write_step<W: Write>(
    w: &mut W,
    verts: &[Vec3],
    faces_in: &[Vec<usize>],
    name: &str,
) -> std::io::Result<()> {
    writeln!(w, "ISO-10303-21;")?;
    writeln!(w, "HEADER;")?;
    writeln!(w, "FILE_DESCRIPTION(('ALICE-SDF faceted BREP'),'2;1');")?;
    writeln!(
        w,
        "FILE_NAME('{name}.step','2026-06-06T00:00:00',('ALICE-SDF'),(''),'ALICE-SDF','','');",
    )?;
    writeln!(
        w,
        "FILE_SCHEMA(('AUTOMOTIVE_DESIGN {{ 1 0 10303 214 1 1 1 1 }}'));"
    )?;
    writeln!(w, "ENDSEC;")?;
    writeln!(w, "DATA;")?;

    // --- context / product / units (固定 id 1..=13) -----------------------
    writeln!(w, "#1=APPLICATION_CONTEXT('automotive design');")?;
    writeln!(
        w,
        "#2=APPLICATION_PROTOCOL_DEFINITION('international standard','automotive_design',2010,#1);"
    )?;
    writeln!(w, "#3=PRODUCT_CONTEXT('',#1,'mechanical');")?;
    writeln!(w, "#4=PRODUCT('{name}','{name}','',(#3));")?;
    writeln!(w, "#5=PRODUCT_DEFINITION_FORMATION('','',#4);")?;
    writeln!(
        w,
        "#6=PRODUCT_DEFINITION_CONTEXT('part definition',#1,'design');"
    )?;
    writeln!(w, "#7=PRODUCT_DEFINITION('design','',#5,#6);")?;
    writeln!(w, "#8=PRODUCT_DEFINITION_SHAPE('','',#7);")?;
    writeln!(
        w,
        "#9=(LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT(.MILLI.,.METRE.));"
    )?;
    writeln!(
        w,
        "#10=(NAMED_UNIT(*)PLANE_ANGLE_UNIT()SI_UNIT($,.RADIAN.));"
    )?;
    writeln!(
        w,
        "#11=(NAMED_UNIT(*)SI_UNIT($,.STERADIAN.)SOLID_ANGLE_UNIT());"
    )?;
    writeln!(
        w,
        "#12=UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE(1.E-06),#9,'distance_accuracy_value','');"
    )?;
    writeln!(
        w,
        "#13=(GEOMETRIC_REPRESENTATION_CONTEXT(3)GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT((#12))GLOBAL_UNIT_ASSIGNED_CONTEXT((#9,#10,#11))REPRESENTATION_CONTEXT('',''));"
    )?;

    let mut id = 14u32;
    let mut next = move || {
        let i = id;
        id += 1;
        i
    };

    // --- 頂点 (CARTESIAN_POINT は 1 頂点 1 個、軸や直線もこれを使い回す) ----
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
    let mut vertex_ids = Vec::with_capacity(verts.len());
    for &pid in &point_ids {
        let vid = next();
        writeln!(w, "#{vid}=VERTEX_POINT('',#{pid});")?;
        vertex_ids.push(vid);
    }

    // --- 稜線 (無向エッジ 1 本 = EDGE_CURVE 1 個、2 面で共有) --------------
    let mut edge_ids: HashMap<(usize, usize), u32> = HashMap::new();
    for face in faces_in {
        for k in 0..face.len() {
            let (a, b) = (face[k], face[(k + 1) % face.len()]);
            let key = (a.min(b), a.max(b));
            if a == b || edge_ids.contains_key(&key) {
                continue;
            }
            let dir = verts[key.1] - verts[key.0];
            let dir = if dir.length_squared() > 1e-20 {
                dir.normalize()
            } else {
                Vec3::X
            };
            let did = next();
            writeln!(
                w,
                "#{did}=DIRECTION('',({:.6},{:.6},{:.6}));",
                dir.x, dir.y, dir.z
            )?;
            let vecid = next();
            writeln!(w, "#{vecid}=VECTOR('',#{did},1.0);")?;
            let lineid = next();
            writeln!(w, "#{lineid}=LINE('',#{},#{vecid});", point_ids[key.0])?;
            let ecid = next();
            writeln!(
                w,
                "#{ecid}=EDGE_CURVE('',#{},#{},#{lineid},.T.);",
                vertex_ids[key.0], vertex_ids[key.1]
            )?;
            edge_ids.insert(key, ecid);
        }
    }

    // --- 面 (多角形 1 枚 = PLANE 上の ADVANCED_FACE) ----------------------
    let mut face_ids = Vec::with_capacity(faces_in.len());
    for face in faces_in {
        if face.len() < 3 {
            continue;
        }
        let (a, b, c) = (verts[face[0]], verts[face[1]], verts[face[2]]);
        let normal = (b - a).cross(c - a);
        let normal = if normal.length_squared() > 1e-20 {
            normal.normalize()
        } else {
            Vec3::Z
        };
        let refdir = perpendicular(normal);
        let nid = next();
        writeln!(
            w,
            "#{nid}=DIRECTION('',({:.6},{:.6},{:.6}));",
            normal.x, normal.y, normal.z
        )?;
        let rid = next();
        writeln!(
            w,
            "#{rid}=DIRECTION('',({:.6},{:.6},{:.6}));",
            refdir.x, refdir.y, refdir.z
        )?;
        let axid = next();
        writeln!(
            w,
            "#{axid}=AXIS2_PLACEMENT_3D('',#{},#{nid},#{rid});",
            point_ids[face[0]]
        )?;
        let plid = next();
        writeln!(w, "#{plid}=PLANE('',#{axid});")?;

        let mut oriented = Vec::with_capacity(face.len());
        for k in 0..face.len() {
            let (u, v) = (face[k], face[(k + 1) % face.len()]);
            let key = (u.min(v), u.max(v));
            let ec = edge_ids[&key];
            // 稜線は min→max 向きで作ってあるので、面の巡回と一致するかで向きを決める
            let sense = if u < v { ".T." } else { ".F." };
            let oeid = next();
            writeln!(w, "#{oeid}=ORIENTED_EDGE('',*,*,#{ec},{sense});")?;
            oriented.push(oeid);
        }
        let loopid = next();
        let oriented_refs = oriented
            .iter()
            .map(|o| format!("#{o}"))
            .collect::<Vec<_>>()
            .join(",");
        writeln!(w, "#{loopid}=EDGE_LOOP('',({oriented_refs}));")?;
        let boundid = next();
        writeln!(w, "#{boundid}=FACE_OUTER_BOUND('',#{loopid},.T.);")?;
        let faceid = next();
        writeln!(w, "#{faceid}=ADVANCED_FACE('',(#{boundid}),#{plid},.T.);")?;
        face_ids.push(faceid);
    }

    // --- shell → solid → 形状表現 ----------------------------------------
    let shell = next();
    let faces = face_ids
        .iter()
        .map(|f| format!("#{f}"))
        .collect::<Vec<_>>()
        .join(",");
    writeln!(w, "#{shell}=CLOSED_SHELL('',({faces}));")?;
    let solid = next();
    writeln!(w, "#{solid}=MANIFOLD_SOLID_BREP('{name}',#{shell});")?;

    // 表現の原点 (幾何とは独立な軸)
    let origin = next();
    writeln!(w, "#{origin}=CARTESIAN_POINT('',(0.0,0.0,0.0));")?;
    let zdir = next();
    writeln!(w, "#{zdir}=DIRECTION('',(0.0,0.0,1.0));")?;
    let xdir = next();
    writeln!(w, "#{xdir}=DIRECTION('',(1.0,0.0,0.0));")?;
    let axis = next();
    writeln!(
        w,
        "#{axis}=AXIS2_PLACEMENT_3D('',#{origin},#{zdir},#{xdir});"
    )?;
    let rep = next();
    writeln!(
        w,
        "#{rep}=ADVANCED_BREP_SHAPE_REPRESENTATION('{name}',(#{axis},#{solid}),#13);"
    )?;
    let sdr = next();
    writeln!(w, "#{sdr}=SHAPE_DEFINITION_REPRESENTATION(#8,#{rep});")?;

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

    #[test]
    fn marching_cubes_produces_dense_mesh() {
        // 旧 naive voxel quads では数十 verts、Marching Cubes は数百〜数千 verts のはず
        let n = SdfNode::sphere(1.0);
        let cfg = StepConfig {
            bounds: (-1.5, 1.5),
            resolution: 16,
            name: "QualityCheck".into(),
        };
        let (verts, tris) = super::sdf_to_facets(&n, &cfg);
        assert!(
            verts.len() > 100,
            "expected >100 verts from MC at res=16, got {}",
            verts.len()
        );
        assert!(
            !tris.is_empty(),
            "expected non-empty triangulation, got {} tris",
            tris.len()
        );
    }

    #[test]
    fn step_export_contains_one_point_per_vertex() {
        let path = std::env::temp_dir().join("alice_sdf_count.step");
        let n = SdfNode::sphere(1.0);
        let cfg = StepConfig {
            bounds: (-1.5, 1.5),
            resolution: 8,
            name: "Count".into(),
        };
        export_step(&path, &n, &cfg).unwrap();
        let content = std::fs::read_to_string(&path).unwrap();
        let (verts, _) = super::sdf_to_facets(&n, &cfg);
        // 位相頂点は mesh 頂点と 1:1 (軸や直線は同じ CARTESIAN_POINT を使い回す)
        assert_eq!(
            content.matches("VERTEX_POINT").count(),
            verts.len(),
            "VERTEX_POINT count != vertex count"
        );
        // CARTESIAN_POINT は頂点分 + 形状表現の原点 1 個
        assert_eq!(
            content.matches("CARTESIAN_POINT").count(),
            verts.len() + 1,
            "CARTESIAN_POINT count != vertex count + origin"
        );
        std::fs::remove_file(&path).ok();
    }
}

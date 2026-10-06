//! Format oracles for the mesh / volume exporters and importers in `io`.
//!
//! Two kinds of check, both independent of the code under test:
//!
//! * **Round trip** — a hand-built mesh whose coordinates are exact in both
//!   binary and short decimal form (0, ±0.5, 1, 1.5, −2.25) is exported and
//!   imported back, and every triangle must come back at the same positions.
//!   Text formats that print with `{}` (shortest round-trip representation)
//!   must be bit exact; formats that print with a fixed precision are held to
//!   that precision (stated per test).
//! * **Layout** — the bytes on disk are decoded here by the format's own
//!   specification (STL ASCII grammar, IGES 5.3 fixed columns, the `.splat`
//!   32-byte record, the MagicaVoxel chunk layout, the `.nanite` and `.abm`
//!   headers, glTF 2.0 accessor bounds and data URI), not by the crate's
//!   reader.
//!
//! Author: Moroya Sakamoto

use alice_sdf::io::abm::{read_abm_header, save_abm, ABM_FLAG_HAS_NORMALS, ABM_MAGIC, ABM_VERSION};
use alice_sdf::io::fbx::{
    export_fbx, fbx_animation_to_timeline, import_fbx, import_fbx_full, FbxAnimClip, FbxAnimCurve,
    FbxConfig,
};
use alice_sdf::io::gltf::{export_gltf_json, GltfConfig};
use alice_sdf::io::iges::{export_iges, IgesConfig};
use alice_sdf::io::nanite::{
    export_nanite, export_nanite_json, export_nanite_with_config, NaniteExportConfig, NANITE_MAGIC,
    NANITE_VERSION,
};
use alice_sdf::io::obj::{export_obj, import_obj, ObjConfig};
use alice_sdf::io::ply::{export_ply, import_ply, PlyConfig};
use alice_sdf::io::splat::{load_splat, save_splat, Splat, SPLAT_BYTES};
use alice_sdf::io::stl::{export_stl_ascii, import_stl};
use alice_sdf::io::usd::{export_usda, import_usda, UsdConfig};
use alice_sdf::io::vox::{load_vox, save_vox, VoxModel, Voxel};
use alice_sdf::io::IoError;
use alice_sdf::material::{Material, MaterialLibrary};
use alice_sdf::mesh::nanite::{generate_nanite_mesh, NaniteConfig};
use alice_sdf::mesh::{Mesh, Vertex};
use alice_sdf::types::SdfNode;
use glam::{Vec2, Vec3};
use std::collections::HashMap;
use std::path::PathBuf;

fn tmp(name: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("alice_sdf_io_format_oracle_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    dir.join(name)
}

/// A closed tetrahedron, outward winding, coordinates exact in decimal.
fn tetra() -> Mesh {
    let p = [
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.5, 0.0, 0.0),
        Vec3::new(0.0, 1.0, -0.5),
        Vec3::new(0.5, -2.25, 1.0),
    ];
    let centroid = (p[0] + p[1] + p[2] + p[3]) / 4.0;
    let mut vertices = Vec::new();
    for (i, &q) in p.iter().enumerate() {
        let mut v = Vertex::new(q, (q - centroid).normalize());
        v.uv = Vec2::new(i as f32 * 0.25, 0.5);
        vertices.push(v);
    }
    // orient every face outward
    let mut indices = Vec::new();
    for f in [[0_u32, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]] {
        let (a, b, c) = (p[f[0] as usize], p[f[1] as usize], p[f[2] as usize]);
        let n = (b - a).cross(c - a);
        let face_centre = (a + b + c) / 3.0;
        if n.dot(face_centre - centroid) > 0.0 {
            indices.extend_from_slice(&f);
        } else {
            indices.extend_from_slice(&[f[0], f[2], f[1]]);
        }
    }
    Mesh { vertices, indices }
}

fn triangles(mesh: &Mesh) -> Vec<[Vec3; 3]> {
    mesh.indices
        .chunks_exact(3)
        .map(|t| {
            [
                mesh.vertices[t[0] as usize].position,
                mesh.vertices[t[1] as usize].position,
                mesh.vertices[t[2] as usize].position,
            ]
        })
        .collect()
}

/// Same triangles in the same order, each vertex within `tol`. Returns the
/// number of compared vertices (asserted non-zero by the callers).
fn assert_same_triangles(what: &str, got: &Mesh, want: &Mesh, tol: f32) -> usize {
    let (g, w) = (triangles(got), triangles(want));
    assert_eq!(g.len(), w.len(), "{what}: triangle count");
    let mut compared = 0;
    for (i, (a, b)) in g.iter().zip(&w).enumerate() {
        for k in 0..3 {
            assert!(
                (a[k] - b[k]).abs().max_element() <= tol,
                "{what}: triangle {i} vertex {k}: {:?} vs {:?}",
                a[k],
                b[k]
            );
            compared += 1;
        }
    }
    compared
}

// ── text mesh formats ───────────────────────────────────────

#[test]
fn stl_ascii_follows_the_grammar_and_round_trips_bit_exact() {
    let mesh = tetra();
    let path = tmp("tetra_ascii.stl");
    export_stl_ascii(&mesh, &path).unwrap();

    // independent parse: `solid`, then per facet `facet normal` + 3 `vertex`
    let text = std::fs::read_to_string(&path).unwrap();
    let lines: Vec<&str> = text.lines().map(str::trim).collect();
    assert!(lines[0].starts_with("solid"));
    assert!(lines.last().unwrap().starts_with("endsolid"));
    let parsed: Vec<Vec3> = lines
        .iter()
        .filter_map(|l| l.strip_prefix("vertex "))
        .map(|l| {
            let v: Vec<f32> = l.split_whitespace().map(|x| x.parse().unwrap()).collect();
            Vec3::new(v[0], v[1], v[2])
        })
        .collect();
    let want: Vec<Vec3> = triangles(&mesh).into_iter().flatten().collect();
    assert_eq!(parsed, want, "STL ASCII vertices");
    assert_eq!(
        lines
            .iter()
            .filter(|l| l.starts_with("facet normal"))
            .count(),
        4
    );

    let back = import_stl(&path).unwrap();
    assert_eq!(assert_same_triangles("STL ASCII", &back, &mesh, 0.0), 12);
}

#[test]
fn obj_and_ply_round_trip_bit_exact() {
    let mesh = tetra();
    let obj = tmp("tetra.obj");
    export_obj(&mesh, &obj, &ObjConfig::default(), None).unwrap();
    assert_eq!(
        assert_same_triangles("OBJ", &import_obj(&obj).unwrap(), &mesh, 0.0),
        12
    );

    let ply = tmp("tetra.ply");
    export_ply(&mesh, &ply, &PlyConfig::default()).unwrap();
    assert_eq!(
        assert_same_triangles("PLY", &import_ply(&ply).unwrap(), &mesh, 0.0),
        12
    );
}

#[test]
fn fbx_ascii_round_trips_and_binary_is_refused_by_name() {
    let mesh = tetra();
    let path = tmp("tetra_ascii.fbx");
    export_fbx(&mesh, &path, &FbxConfig::default(), None).unwrap();
    assert_eq!(
        assert_same_triangles("FBX ASCII", &import_fbx(&path).unwrap(), &mesh, 1e-6),
        12
    );
    let full = import_fbx_full(&path).unwrap();
    assert_eq!(
        assert_same_triangles("FBX ASCII full", &full.mesh, &mesh, 1e-6),
        12
    );
    // a static mesh carries no skeleton and no animation
    assert!(full.skeleton.is_none());
    assert!(full.animations.is_empty());

    // binary FBX: the file starts with the Kaydara magic (FBX binary header),
    // and import refuses it as an unsupported format rather than an I/O error
    let bin = tmp("tetra_binary.fbx");
    export_fbx(&mesh, &bin, &FbxConfig::binary(), None).unwrap();
    let bytes = std::fs::read(&bin).unwrap();
    assert_eq!(&bytes[..21], b"Kaydara FBX Binary  \0");
    let err = import_fbx(&bin).unwrap_err();
    assert!(matches!(err, IoError::InvalidFormat(_)), "{err}");
    assert!(matches!(
        import_fbx_full(&bin).unwrap_err(),
        IoError::InvalidFormat(_)
    ));
}

#[test]
fn fbx_animation_maps_root_channels_to_timeline_tracks() {
    // oracle: translation is copied, rotation goes from degrees to radians,
    // and keys interpolate linearly
    let mut curves = HashMap::new();
    curves.insert(
        "root".to_string(),
        vec![
            FbxAnimCurve {
                property: "Lcl Translation".to_string(),
                channel: 1,
                times: vec![0.0, 2.0],
                values: vec![1.0, 5.0],
            },
            FbxAnimCurve {
                property: "Lcl Rotation".to_string(),
                channel: 2,
                times: vec![0.0, 2.0],
                values: vec![0.0, 180.0],
            },
            FbxAnimCurve {
                property: "Lcl Scaling".to_string(),
                channel: 0,
                times: vec![0.0],
                values: vec![3.0],
            },
        ],
    );
    curves.insert(
        "other".to_string(),
        vec![FbxAnimCurve {
            property: "Lcl Translation".to_string(),
            channel: 0,
            times: vec![0.0],
            values: vec![9.0],
        }],
    );
    let clip = FbxAnimClip {
        name: "walk".to_string(),
        bone_curves: curves,
        duration: 2.0,
        fps: 30.0,
    };
    let tl = fbx_animation_to_timeline(&clip, "root");
    let at = |track: &str, t: f32| tl.get_value(track, t).unwrap_or(f32::NAN);
    assert!((at("translate.y", 0.5) - 2.0).abs() < 1e-6);
    assert!((at("rotate.z", 1.0) - std::f32::consts::FRAC_PI_2).abs() < 1e-6);
    assert!((at("rotate.z", 2.0) - std::f32::consts::PI).abs() < 1e-6);
    // scaling is not mapped, and other bones do not leak in
    let names: Vec<&str> = tl.evaluate(0.0).into_iter().map(|(n, _)| n).collect();
    assert_eq!(names.len(), 2, "tracks: {names:?}");
    assert!(tl.get_value("translate.x", 0.0).is_none());
}

#[test]
fn usda_round_trips_the_mesh_and_the_material() {
    let mesh = tetra();
    let lib = MaterialLibrary {
        materials: vec![Material {
            base_color: [0.75, 0.25, 0.5, 1.0],
            metallic: 0.5,
            roughness: 0.125,
            opacity: 0.875,
            ..Default::default()
        }],
    };
    let path = tmp("tetra.usda");
    export_usda(&mesh, &path, &UsdConfig::default(), Some(&lib)).unwrap();
    let back = import_usda(&path).unwrap();
    assert_eq!(assert_same_triangles("USDA", &back.mesh, &mesh, 0.0), 12);
    let m = back.material.expect("material exported");
    assert_eq!(m.diffuse_color, [0.75, 0.25, 0.5]);
    assert_eq!((m.metallic, m.roughness, m.opacity), (0.5, 0.125, 0.875));
}

// ── glTF JSON ───────────────────────────────────────────────

fn base64_decode(s: &str) -> Vec<u8> {
    let val = |c: u8| -> u32 {
        match c {
            b'A'..=b'Z' => u32::from(c - b'A'),
            b'a'..=b'z' => u32::from(c - b'a') + 26,
            b'0'..=b'9' => u32::from(c - b'0') + 52,
            b'+' => 62,
            b'/' => 63,
            _ => panic!("not base64: {c}"),
        }
    };
    let mut out = Vec::new();
    for chunk in s.as_bytes().chunks(4) {
        let pad = chunk.iter().rev().take_while(|&&c| c == b'=').count();
        let mut n = 0_u32;
        for &c in chunk {
            n = (n << 6) | if c == b'=' { 0 } else { val(c) };
        }
        let bytes = [(n >> 16) as u8, (n >> 8) as u8, n as u8];
        out.extend_from_slice(&bytes[..3 - pad]);
    }
    out
}

#[test]
fn gltf_json_carries_its_buffer_and_true_position_bounds() {
    let mesh = tetra();
    let path = tmp("tetra.gltf");
    export_gltf_json(&mesh, &path, &GltfConfig::aaa(), None).unwrap();
    let doc: serde_json::Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    assert_eq!(doc["asset"]["version"], "2.0");

    // glTF 2.0 §3.6.1.1: a .gltf buffer needs a uri; here a base64 data URI
    // of exactly byteLength bytes
    let buffer = &doc["buffers"][0];
    let uri = buffer["uri"].as_str().expect("buffers[0].uri");
    let b64 = uri
        .strip_prefix("data:application/octet-stream;base64,")
        .expect("data URI");
    let bin = base64_decode(b64);
    assert_eq!(bin.len() as u64, buffer["byteLength"].as_u64().unwrap());

    // §5.1.1: POSITION accessor min / max are the true component bounds,
    // and the float data in the buffer view are the mesh positions
    let prim = &doc["meshes"][0]["primitives"][0];
    let acc = &doc["accessors"][prim["attributes"]["POSITION"].as_u64().unwrap() as usize];
    assert_eq!(acc["count"].as_u64().unwrap(), 4);
    let mins: Vec<f64> = acc["min"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    let maxs: Vec<f64> = acc["max"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    assert_eq!(mins, vec![0.0, -2.25, -0.5]);
    assert_eq!(maxs, vec![1.5, 1.0, 1.0]);
    let view = &doc["bufferViews"][acc["bufferView"].as_u64().unwrap() as usize];
    let off = view["byteOffset"].as_u64().unwrap_or(0) as usize;
    let mut compared = 0;
    for (i, v) in mesh.vertices.iter().enumerate() {
        for k in 0..3 {
            let at = off + (i * 3 + k) * 4;
            let f = f32::from_le_bytes(bin[at..at + 4].try_into().unwrap());
            assert_eq!(f, v.position[k], "vertex {i} component {k}");
            compared += 1;
        }
    }
    assert_eq!(compared, 12);
}

// ── IGES ────────────────────────────────────────────────────

#[test]
fn iges_follows_the_fixed_column_layout_and_points_both_ways() {
    let path = tmp("sphere.igs");
    let cfg = IgesConfig {
        bounds: (-1.5, 1.5),
        resolution: 8,
    };
    export_iges(&path, &SdfNode::sphere(1.0), &cfg).unwrap();
    let text = std::fs::read_to_string(&path).unwrap();
    let lines: Vec<&str> = text.lines().collect();

    // every record is 80 columns: 72 data, section letter, 7-digit sequence
    let mut by_section: HashMap<char, Vec<&str>> = HashMap::new();
    for l in &lines {
        assert_eq!(l.len(), 80, "record length: {l:?}");
        let sec = l.as_bytes()[72] as char;
        let seq: u32 = l[73..].trim().parse().unwrap();
        let list = by_section.entry(sec).or_default();
        if sec != 'T' {
            assert_eq!(seq as usize, list.len() + 1, "{sec} sequence");
        }
        list.push(l);
    }
    // the terminate record counts each section
    let t = by_section[&'T'][0];
    for (i, sec) in ['S', 'G', 'D', 'P'].iter().enumerate() {
        assert_eq!(t.as_bytes()[i * 8] as char, *sec);
        let n: usize = t[i * 8 + 1..i * 8 + 8].trim().parse().unwrap();
        assert_eq!(n, by_section[sec].len(), "T count for {sec}");
    }

    let d = &by_section[&'D'];
    let p = &by_section[&'P'];
    assert_eq!(d.len(), 2 * p.len(), "two DE lines per entity");
    assert!(!p.is_empty());
    let field = |l: &str, k: usize| -> u32 { l[k * 8..k * 8 + 8].trim().parse().unwrap() };
    let mut nodes = Vec::new();
    let mut compared = 0;
    for (n, pl) in p.iter().enumerate() {
        let de_seq = 2 * n as u32 + 1;
        let de1 = d[2 * n];
        // DE field 2: parameter data pointer = this entity's first P line
        let p_seq: u32 = pl[73..].trim().parse().unwrap();
        assert_eq!(field(de1, 1), p_seq, "DE {de_seq} → P");
        // P col 65-72: back pointer = this entity's DE sequence (IGES 5.3 §2.2.4.5)
        let back: u32 = pl[64..72].trim().parse().unwrap();
        assert_eq!(back, de_seq, "P {p_seq} → DE");
        let data = pl[..64].trim().trim_end_matches(';');
        let params: Vec<&str> = data.split(',').collect();
        assert_eq!(field(de1, 0).to_string(), params[0], "entity type");
        match params[0] {
            "134" => nodes.push((
                de_seq,
                Vec3::new(
                    params[1].parse().unwrap(),
                    params[2].parse().unwrap(),
                    params[3].parse().unwrap(),
                ),
            )),
            "136" => {
                // element nodes reference node DEs
                for r in &params[2..5] {
                    let r: u32 = r.parse().unwrap();
                    assert!(nodes.iter().any(|(s, _)| *s == r), "element → node DE {r}");
                }
            }
            other => panic!("unexpected entity {other}"),
        }
        compared += 1;
    }
    assert_eq!(compared, p.len());
    // nodes are on the unit sphere up to the mesher's error; printed with 6 decimals
    assert!(nodes.len() > 20);
    for (_, q) in &nodes {
        assert!((q.length() - 1.0).abs() < 0.05, "{q:?}");
    }
}

// ── binary formats ──────────────────────────────────────────

#[test]
fn splat_records_are_32_little_endian_bytes() {
    let splats = vec![
        Splat {
            position: [1.5, -2.25, 0.5],
            scale: [0.125, 0.25, 1.0],
            color: [10, 20, 30, 255],
            rotation: [128, 0, 64, 255],
        },
        Splat {
            position: [0.0, 1.0, -1.0],
            scale: [2.0, 2.0, 2.0],
            color: [1, 2, 3, 4],
            rotation: [5, 6, 7, 8],
        },
    ];
    let path = tmp("two.splat");
    save_splat(&path, &splats).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(SPLAT_BYTES, 32);
    assert_eq!(bytes.len(), 2 * 32);
    for (i, s) in splats.iter().enumerate() {
        let r = &bytes[i * 32..(i + 1) * 32];
        let f = |o: usize| f32::from_le_bytes(r[o..o + 4].try_into().unwrap());
        assert_eq!([f(0), f(4), f(8)], s.position);
        assert_eq!([f(12), f(16), f(20)], s.scale);
        assert_eq!(&r[24..28], &s.color);
        assert_eq!(&r[28..32], &s.rotation);
        assert_eq!(&s.to_bytes()[..], r);
    }
    let back = load_splat(&path).unwrap();
    assert_eq!(back.len(), 2);
    for (a, b) in back.iter().zip(&splats) {
        assert_eq!(
            (a.position, a.scale, a.color, a.rotation),
            (b.position, b.scale, b.color, b.rotation)
        );
        assert_eq!(Splat::from_bytes(&b.to_bytes()).position, b.position);
    }
    // a truncated file is rejected, not half-read
    std::fs::write(&path, &bytes[..40]).unwrap();
    assert!(load_splat(&path).is_err());
}

#[test]
fn vox_follows_the_magicavoxel_chunk_layout() {
    let model = VoxModel {
        size: (4, 5, 6),
        voxels: vec![
            Voxel {
                x: 0,
                y: 1,
                z: 2,
                color: 7,
            },
            Voxel {
                x: 3,
                y: 4,
                z: 5,
                color: 255,
            },
        ],
    };
    let path = tmp("m.vox");
    save_vox(&path, &model).unwrap();
    let b = std::fs::read(&path).unwrap();
    let u32_at = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap());
    // "VOX " + version 150, MAIN (content 0, children = rest of file)
    assert_eq!(&b[0..4], b"VOX ");
    assert_eq!(u32_at(4), 150);
    assert_eq!(&b[8..12], b"MAIN");
    assert_eq!(u32_at(12), 0);
    assert_eq!(u32_at(16) as usize, b.len() - 20);
    // SIZE: 12 bytes x, y, z
    assert_eq!(&b[20..24], b"SIZE");
    assert_eq!(
        (u32_at(24), u32_at(32), u32_at(36), u32_at(40)),
        (12, 4, 5, 6)
    );
    // XYZI: count, then x y z colour per voxel
    assert_eq!(&b[44..48], b"XYZI");
    assert_eq!(u32_at(48), 4 + 2 * 4);
    assert_eq!(u32_at(56), 2);
    assert_eq!(&b[60..68], &[0, 1, 2, 7, 3, 4, 5, 255]);
    assert_eq!(b.len(), 68);

    let back = load_vox(&path).unwrap();
    assert_eq!(back.size, model.size);
    let v: Vec<(u8, u8, u8, u8)> = back
        .voxels
        .iter()
        .map(|v| (v.x, v.y, v.z, v.color))
        .collect();
    assert_eq!(v, vec![(0, 1, 2, 7), (3, 4, 5, 255)]);
}

#[test]
fn abm_header_states_the_counts_of_the_saved_mesh() {
    let mesh = tetra();
    let path = tmp("tetra.abm");
    save_abm(&mesh, &path).unwrap();
    let h = read_abm_header(&path).unwrap();
    assert_eq!(h.magic, ABM_MAGIC);
    assert_eq!(h.version, ABM_VERSION);
    assert_eq!(h.vertex_count, 4);
    assert_eq!(h.index_count, 12);
    assert_eq!(h.lod_count, 0);
    assert_ne!(h.flags & ABM_FLAG_HAS_NORMALS, 0);
    // the header is the first 32 bytes, little endian (magic, version, flags, counts)
    let b = std::fs::read(&path).unwrap();
    assert_eq!(&b[0..8], b"ALICEBM\0");
    assert_eq!(u16::from_le_bytes([b[8], b[9]]), ABM_VERSION);
    assert_eq!(u32::from_le_bytes(b[12..16].try_into().unwrap()), 4);
    assert_eq!(u32::from_le_bytes(b[16..20].try_into().unwrap()), 12);
    // a file that is not ABM is rejected
    std::fs::write(&path, [0_u8; 32]).unwrap();
    assert!(read_abm_header(&path).is_err());
}

#[test]
fn nanite_file_size_and_header_follow_the_layout() {
    let nanite = generate_nanite_mesh(
        &SdfNode::sphere(1.0),
        Vec3::splat(-1.5),
        Vec3::splat(1.5),
        &NaniteConfig {
            lod_levels: 2,
            base_resolution: 16,
            ..NaniteConfig::default()
        },
    );
    assert!(!nanite.clusters.is_empty());
    let verts: usize = nanite.clusters.iter().map(|c| c.vertices.len()).sum();
    let tris: usize = nanite.clusters.iter().map(|c| c.triangles.len()).sum();
    let dag: usize = nanite
        .clusters
        .iter()
        .map(|c| c.parent_ids.len() + c.child_ids.len())
        .sum();

    // layout: 32-byte header, 16 bytes per LOD, per cluster 16 (ids / counts)
    // + 40 (bounds) + 8 (error, material) + 8 (DAG counts) + 4 per DAG id,
    // then per vertex 12 + 12 (normals) + 8 (uvs), per triangle 12
    let expected = |normals: bool, uvs: bool| {
        32 + 16 * nanite.lod_levels.len()
            + 72 * nanite.clusters.len()
            + 4 * dag
            + verts * (12 + if normals { 12 } else { 0 } + if uvs { 8 } else { 0 })
            + 12 * tris
    };
    let path = tmp("sphere.nanite");
    export_nanite(&nanite, &path).unwrap();
    let b = std::fs::read(&path).unwrap();
    assert_eq!(b.len(), expected(true, true));
    let u32_at = |b: &[u8], o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap());
    assert_eq!(&b[0..4], NANITE_MAGIC);
    assert_eq!(u32_at(&b, 4), NANITE_VERSION);
    assert_eq!(u32_at(&b, 8) as usize, nanite.clusters.len());
    assert_eq!(u32_at(&b, 12) as usize, nanite.groups.len());
    assert_eq!(u32_at(&b, 16) as usize, nanite.lod_levels.len());
    assert_eq!(u32_at(&b, 20) as usize, verts);
    assert_eq!(u32_at(&b, 24) as usize, tris);
    assert_eq!(u32_at(&b, 28), 0b011);

    let lean = NaniteExportConfig {
        export_uvs: false,
        export_normals: false,
        quantize_positions: false,
    };
    export_nanite_with_config(&nanite, &path, &lean).unwrap();
    let b = std::fs::read(&path).unwrap();
    assert_eq!(b.len(), expected(false, false));
    assert_eq!(u32_at(&b, 28), 0);

    let json = tmp("sphere.nanite.json");
    export_nanite_json(&nanite, &json).unwrap();
    let doc: serde_json::Value = serde_json::from_slice(&std::fs::read(&json).unwrap()).unwrap();
    assert_eq!(
        doc["format_version"].as_u64().unwrap(),
        u64::from(NANITE_VERSION)
    );
    assert_eq!(
        doc["cluster_count"].as_u64().unwrap() as usize,
        nanite.clusters.len()
    );
    assert_eq!(
        doc["group_count"].as_u64().unwrap() as usize,
        nanite.groups.len()
    );
}

#[cfg(feature = "hlsl")]
#[test]
fn nanite_material_is_a_hlsl_function_of_the_tree() {
    use alice_sdf::io::nanite::export_nanite_hlsl_material;
    let path = tmp("sphere_material.ush");
    export_nanite_hlsl_material(&SdfNode::sphere(1.0), &path).unwrap();
    let text = std::fs::read_to_string(&path).unwrap();
    assert!(text.contains("float"), "HLSL body expected");
    assert!(text.lines().count() > 3);
}

#[cfg(feature = "openvdb")]
#[test]
fn vdb_grid_holds_the_sphere_distance_at_every_node() {
    use alice_sdf::io::vdb::{
        bake_dense_grid, bake_to_vdb, load_dense_grid_from_vdb, DenseGrid, VdbError,
    };
    let (res, bounds) = (9_u32, (-2.0_f32, 2.0_f32));
    let bytes = bake_to_vdb(&SdfNode::sphere(1.0), bounds, res).unwrap();
    let (values, r, b): DenseGrid = load_dense_grid_from_vdb(&bytes).unwrap();
    assert_eq!((r, b), (res, bounds));
    assert_eq!(values.len(), 729);
    // x fastest: index i + n·(j + n·k), node coordinate min + index · step
    let step = (bounds.1 - bounds.0) / (res - 1) as f32;
    let mut compared = 0;
    for k in 0..res {
        for j in 0..res {
            for i in 0..res {
                let p = Vec3::new(i as f32, j as f32, k as f32) * step + Vec3::splat(bounds.0);
                let idx = (i + res * (j + res * k)) as usize;
                assert!((values[idx] - (p.length() - 1.0)).abs() < 1e-5, "{p:?}");
                compared += 1;
            }
        }
    }
    assert_eq!(compared, 729);
    assert_eq!(bake_dense_grid(&SdfNode::sphere(1.0), bounds, res), values);
    assert!(matches!(
        bake_to_vdb(&SdfNode::sphere(1.0), (1.0, 1.0), 4),
        Err(VdbError::InvalidBounds)
    ));
}

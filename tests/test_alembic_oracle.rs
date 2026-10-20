//! The `.abc` writer against the Alembic library (ALICE-Law G8 and the format claim)
//!
//! oracle: tests/oracle/alembic_reader.cpp reads the archive with the Alembic library (not this
//! crate) and prints each fact bit for bit; tests/oracle/alembic_refwriter.cpp writes the same mesh
//! with the library. The test compares what the library reads from this crate's file with the mesh
//! itself and with what it reads from the library's own file for that mesh (object and property
//! hashes, stored and recomputed sample digests).
//!
//! The two programs are built by the `alembic-oracle` CI job (Linux, against the Alembic release in
//! tests/oracle/alembic-library.txt, built from its tag tarball), which sets
//! `ALICE_SDF_ALEMBIC_READER` / `ALICE_SDF_ALEMBIC_REFWRITER` and runs the ignored tests; a missing
//! program fails, so the job cannot pass having compared nothing.
//!
//! Author: Moroya Sakamoto

use alice_sdf::io::{export_alembic, export_alembic_with_provenance, AlembicConfig};
use alice_sdf::mesh::{sdf_to_mesh, MarchingCubesConfig, Mesh, Vertex};
use alice_sdf::prelude::*;
use alice_sdf::provenance::Provenance;
use std::path::{Path, PathBuf};
use std::process::Command;

fn dir() -> PathBuf {
    let d = std::env::temp_dir().join(format!("alice_sdf_alembic_oracle_{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn program(var: &str) -> PathBuf {
    let p = std::env::var_os(var).unwrap_or_else(|| {
        panic!("{var} is not set (build tests/oracle/*.cpp against the Alembic library)")
    });
    PathBuf::from(p)
}

/// The reader's lines for `path`, or its error output when the library does not open it
fn read(path: &Path) -> Result<Vec<String>, String> {
    let out = Command::new(program("ALICE_SDF_ALEMBIC_READER"))
        .arg(path)
        .output()
        .unwrap();
    if out.status.success() {
        Ok(String::from_utf8(out.stdout)
            .unwrap()
            .lines()
            .map(str::to_string)
            .collect())
    } else {
        Err(String::from_utf8_lossy(&out.stderr).into_owned())
    }
}

fn mesh() -> Mesh {
    let shape = SdfNode::sphere(1.0).subtract(SdfNode::box3d(0.6, 0.6, 0.6));
    sdf_to_mesh(
        &shape,
        Vec3::splat(-1.5),
        Vec3::splat(1.5),
        &MarchingCubesConfig {
            resolution: 8,
            compute_normals: true,
            ..Default::default()
        },
    )
}

/// One triangle whose bounds tie 0.0 against -0.0: x has 0.0 first and -0.0 later at its
/// maximum, y has -0.0 first and 0.0 later at its minimum. The library's bounds (Imath
/// Box::extendBy, a strict comparison in vertex order) keep the first of each tie, so a writer that
/// takes min / max or a non-strict comparison stores other bits in `.selfBnds`
fn signed_zero_tie() -> Mesh {
    let v = |x: f32, y: f32, z: f32| Vertex {
        position: Vec3::new(x, y, z),
        normal: Vec3::Z,
        uv: glam::Vec2::new(x, y),
        uv2: glam::Vec2::ZERO,
        tangent: glam::Vec4::ZERO,
        color: [1.0; 4],
        material_id: 0,
    };
    Mesh {
        vertices: vec![v(0.0, -0.0, 0.0), v(-0.0, 0.0, -0.0), v(-1.0, 1.0, 0.0)],
        indices: vec![0, 1, 2],
    }
}

fn hex(v: &[f32]) -> String {
    let mut s = v.len().to_string();
    for f in v {
        s.push_str(&format!(" {:08x}", f.to_bits()));
    }
    s
}

fn ints(v: &[i32]) -> String {
    let mut s = v.len().to_string();
    for i in v {
        s.push_str(&format!(" {i}"));
    }
    s
}

/// What the Alembic library must read from the archive of `m` (Alembic stores faces clockwise, so
/// each triangle's indices are reversed)
fn expected(m: &Mesh, config: &AlembicConfig) -> Vec<String> {
    let p: Vec<f32> = m
        .vertices
        .iter()
        .flat_map(|v| v.position.to_array())
        .collect();
    let fi: Vec<i32> = m
        .indices
        .chunks(3)
        .flat_map(|t| [t[0] as i32, t[2] as i32, t[1] as i32])
        .collect();
    let fc = vec![3i32; m.indices.len() / 3];
    let mut out = vec![
        format!("P {}", hex(&p)),
        format!("faceIndices {}", ints(&fi)),
        format!("faceCounts {}", ints(&fc)),
    ];
    if config.export_normals {
        let n: Vec<f32> = m
            .vertices
            .iter()
            .flat_map(|v| v.normal.to_array())
            .collect();
        out.push(format!("N {}", hex(&n)));
    }
    if config.export_uvs {
        let uv: Vec<f32> = m.vertices.iter().flat_map(|v| v.uv.to_array()).collect();
        out.push(format!("uv {}", hex(&uv)));
    }
    out
}

fn line<'a>(lines: &'a [String], tag: &str) -> Vec<&'a String> {
    lines
        .iter()
        .filter(|l| l.starts_with(&format!("{tag} ")))
        .collect()
}

const CONFIGS: [(bool, bool); 4] = [(true, true), (true, false), (false, true), (false, false)];

#[test]
#[ignore = "needs the Alembic library programs of tests/oracle (the alembic-oracle CI job)"]
fn the_alembic_library_reads_the_mesh_and_the_identifier_bit_for_bit() {
    let m = mesh();
    assert!(m.indices.len() >= 3 * 16, "a mesh with faces to compare");
    let mut compared = 0;
    for (normals, uvs) in CONFIGS {
        let config = AlembicConfig {
            export_normals: normals,
            export_uvs: uvs,
            ..AlembicConfig::default()
        };
        let path = dir().join(format!("m_{normals}_{uvs}.abc"));
        export_alembic(&m, &path, &config).unwrap();
        let got =
            read(&path).unwrap_or_else(|e| panic!("the Alembic library does not open it: {e}"));
        assert_eq!(line(&got, "archive_valid"), vec!["archive_valid 1"]);
        assert_eq!(
            line(&got, "meta alice:semantics_id"),
            vec![&format!(
                "meta alice:semantics_id {}",
                Provenance::sdf().hex()
            )]
        );
        assert_eq!(line(&got, "polymesh"), vec!["polymesh /mesh"]);
        for want in expected(&m, &config) {
            let tag = want.split(' ').next().unwrap();
            assert_eq!(line(&got, tag), vec![&want], "{tag} ({normals}, {uvs})");
            compared += 1;
        }
        for d in line(&got, "digest") {
            let f: Vec<&str> = d.split(' ').collect();
            assert_eq!(f[2], f[3], "stored digest is not the library's: {d}");
        }
    }
    assert_eq!(compared, 4 * 3 + 2 + 2);
}

#[test]
#[ignore = "needs the Alembic library programs of tests/oracle (the alembic-oracle CI job)"]
fn this_crate_writes_what_the_alembic_library_writes_for_the_same_mesh() {
    let id = Provenance::sdf();
    let meshes = [
        ("mc", mesh()),
        ("tie", signed_zero_tie()),
        ("empty", Mesh::new()),
    ];
    for ((name, m), (normals, uvs)) in meshes
        .iter()
        .flat_map(|m| CONFIGS.iter().map(move |c| (m, *c)))
    {
        let config = AlembicConfig {
            export_normals: normals,
            export_uvs: uvs,
            ..AlembicConfig::default()
        };
        let ours = dir().join(format!("ours_{name}_{normals}_{uvs}.abc"));
        export_alembic_with_provenance(m, &ours, &config, &id).unwrap();
        let input = dir().join(format!("in_{name}_{normals}_{uvs}.txt"));
        let mut text = format!("meta alice:semantics_id {}\n", id.hex());
        for l in expected(m, &config) {
            text.push_str(&l);
            text.push('\n');
        }
        std::fs::write(&input, text).unwrap();
        let theirs = dir().join(format!("lib_{name}_{normals}_{uvs}.abc"));
        let st = Command::new(program("ALICE_SDF_ALEMBIC_REFWRITER"))
            .arg(&input)
            .arg(&theirs)
            .status()
            .unwrap();
        assert!(st.success(), "the reference writer failed");
        // the library stamps its own version string; everything else must agree
        let keep = |v: Vec<String>| -> Vec<String> {
            v.into_iter()
                .filter(|l| !l.starts_with("meta _ai_"))
                .collect()
        };
        let a = keep(read(&ours).unwrap_or_else(|e| panic!("ours: {e}")));
        let b = keep(read(&theirs).unwrap_or_else(|e| panic!("library: {e}")));
        assert!(b.iter().any(|l| l.starts_with("hash /mesh ")), "{b:?}");
        assert_eq!(a, b, "{name} ({normals}, {uvs})");
    }
}

#[test]
#[ignore = "needs the Alembic library programs of tests/oracle (the alembic-oracle CI job)"]
fn the_alembic_library_refuses_a_broken_header() {
    let path = dir().join("good.abc");
    export_alembic(&mesh(), &path, &AlembicConfig::default()).unwrap();
    let good = std::fs::read(&path).unwrap();
    read(&path).unwrap_or_else(|e| panic!("the unbroken file must open: {e}"));
    // an archive the writer never finished (frozen byte 0x00), and another format version
    for (at, value) in [(5usize, 0x00u8), (7, 0x02)] {
        let mut b = good.clone();
        b[at] = value;
        let p = dir().join(format!("broken_{at}.abc"));
        std::fs::write(&p, &b).unwrap();
        assert!(read(&p).is_err(), "byte {at} = {value:#04x} still opens");
    }
}

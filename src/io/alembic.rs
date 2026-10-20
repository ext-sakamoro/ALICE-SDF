//! Alembic (.abc) Export
//!
//! Writes a mesh as an Alembic archive in the Ogawa layout (the layout of Alembic 1.8): one
//! static `PolyMesh` object named `mesh` under the top object, with positions `P`,
//! `.faceIndices`, `.faceCounts`, `.selfBnds` and, when configured, per-vertex normals `N` and
//! UVs `uv`. The Alembic library reads it (tests/test_alembic_oracle.rs reads it with the library
//! and compares it with what the library itself writes for the same mesh), so the tools built on
//! it (Maya, Houdini, Blender, Nuke, Katana) can open it.
//!
//! Faces are stored clockwise, as Alembic stores them: each triangle `(a, b, c)` of the mesh is
//! written as `(a, c, b)`. The archive metadata holds `_ai_Application`, `_ai_DCC_FPS` and the
//! semantics identifier as `alice:semantics_id=<64 hex>` ([`crate::provenance`]).
//!
//! The layout written here: the Ogawa header (`Ogawa`, the frozen byte `0xFF`, version `00 01`,
//! the `u64` position of the root group); a group is a `u64` child count and the `u64` child
//! positions (bit 63 marks a data child), a data block is a `u64` length and the bytes, all
//! little-endian. On top of it, the AbcCoreOgawa structure: object and property headers, the
//! 16-byte digest before each sample (MurmurHash3 x64 128 of its bytes), the object and property
//! hashes (SpookyHash V2), the time sampling and the indexed metadata.
//!
//! Author: Moroya Sakamoto

use crate::io::IoError;
use crate::mesh::Mesh;
use crate::provenance::{Provenance, KEY};
use std::path::Path;

/// Configuration for Alembic export
#[derive(Debug, Clone)]
pub struct AlembicConfig {
    /// Frames per second, stored as the archive metadata `_ai_DCC_FPS` (the mesh is static)
    pub fps: f64,
    /// Export vertex normals (`N`, per vertex)
    pub export_normals: bool,
    /// Export UV coordinates (`uv`, per vertex)
    pub export_uvs: bool,
}

impl Default for AlembicConfig {
    fn default() -> Self {
        Self {
            fps: 24.0,
            export_normals: true,
            export_uvs: true,
        }
    }
}

/// Export a mesh as an Alembic (Ogawa) archive with this crate's semantics identifier in the
/// archive metadata
pub fn export_alembic(
    mesh: &Mesh,
    path: impl AsRef<Path>,
    config: &AlembicConfig,
) -> Result<(), IoError> {
    export_alembic_with_provenance(mesh, path, config, &Provenance::sdf())
}

/// Export a mesh as an Alembic (Ogawa) archive with the caller's semantics identifier in the
/// archive metadata
pub fn export_alembic_with_provenance(
    mesh: &Mesh,
    path: impl AsRef<Path>,
    config: &AlembicConfig,
    provenance: &Provenance,
) -> Result<(), IoError> {
    std::fs::write(path, archive_bytes(mesh, config, provenance))?;
    Ok(())
}

/// The version number of the archive layout (Alembic's `ALEMBIC_OGAWA_FILE_VERSION`)
const OGAWA_FILE_VERSION: i32 = 0;
/// The Alembic library version whose layout this follows (1.8.12), stored where the library
/// stores its own
const LIBRARY_VERSION: i32 = 10812;
/// A data child with no bytes
const EMPTY_DATA: u64 = 1 << 63;
/// A group with no children
const EMPTY_GROUP: u64 = 0;

// plain old data types, as Alembic numbers them
const POD_INT32: u8 = 6;
const POD_FLOAT32: u8 = 10;
const POD_FLOAT64: u8 = 11;

const MD_BOX: &str = "interpretation=box";
const MD_POINT: &str = "geoScope=vtx;interpretation=point";
const MD_UV: &str = "arrayExtent=1;geoScope=vtx;interpretation=vector;isGeomParam=true;podExtent=2;podName=float32_t";
const MD_NORMAL: &str = "arrayExtent=1;geoScope=vtx;interpretation=normal;isGeomParam=true;podExtent=3;podName=float32_t";
const MD_GEOM: &str = "schema=AbcGeom_PolyMesh_v1;schemaBaseType=AbcGeom_GeomBase_v1";
const MD_OBJECT: &str =
    "schema=AbcGeom_PolyMesh_v1;schemaBaseType=AbcGeom_GeomBase_v1;schemaObjTitle=AbcGeom_PolyMesh_v1:.geom";

/// The Ogawa layer: blocks are appended, a group is written once its children are
struct Ogawa(Vec<u8>);

impl Ogawa {
    fn new() -> Self {
        // magic, the frozen byte (set when finished), version 0.1, root position (set when finished)
        let mut b = b"Ogawa\x00\x00\x01".to_vec();
        b.extend_from_slice(&[0; 8]);
        Self(b)
    }

    fn data(&mut self, parts: &[&[u8]]) -> u64 {
        let len: usize = parts.iter().map(|p| p.len()).sum();
        if len == 0 {
            return EMPTY_DATA;
        }
        let at = self.0.len() as u64;
        self.0.extend_from_slice(&(len as u64).to_le_bytes());
        for p in parts {
            self.0.extend_from_slice(p);
        }
        at | EMPTY_DATA
    }

    fn group(&mut self, children: &[u64]) -> u64 {
        if children.is_empty() {
            return EMPTY_GROUP;
        }
        let at = self.0.len() as u64;
        self.0
            .extend_from_slice(&(children.len() as u64).to_le_bytes());
        for c in children {
            self.0.extend_from_slice(&c.to_le_bytes());
        }
        at
    }

    fn finish(mut self, root: u64) -> Vec<u8> {
        self.0[8..16].copy_from_slice(&root.to_le_bytes());
        self.0[5] = 0xFF;
        self.0
    }
}

/// Metadata strings shared through the archive's index (1-based in the headers, 0 = empty)
#[derive(Default)]
struct MetaDataMap(Vec<&'static str>);

impl MetaDataMap {
    fn index(&mut self, s: &'static str) -> u8 {
        if s.is_empty() {
            return 0;
        }
        let i = match self.0.iter().position(|x| *x == s) {
            Some(i) => i,
            None => {
                self.0.push(s);
                self.0.len() - 1
            }
        };
        // at most 7 strings here, all shorter than 256 bytes: the index never reaches 0xFF
        u8::try_from(i + 1).unwrap_or(0xFF)
    }

    fn bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        for s in &self.0 {
            out.push(s.len() as u8);
            out.extend_from_slice(s.as_bytes());
        }
        out
    }
}

/// A property of the `.geom` compound, as its header and hash need it
struct Prop {
    name: &'static str,
    meta: &'static str,
    scalar: bool,
    pod: u8,
    extent: u8,
    points: usize,
}

impl Prop {
    /// The property header bytes (AbcCoreOgawa `WritePropertyInfo`, one sample, every size
    /// below 256 so one byte each)
    fn info(&self, map: &mut MetaDataMap) -> Vec<u8> {
        let index = map.index(self.meta);
        let mut info: u32 = u32::from(index) << 20;
        info |= if self.scalar { 1 } else { 2 };
        if self.scalar || self.points == 1 {
            info |= 1; // scalar-like
        }
        info |= u32::from(self.pod) << 4;
        info |= 0x800; // one sample: first and last changed index 0
        info |= u32::from(self.extent) << 12;
        // homogenous: a scalar always; an array when its point count equals the pod count of the
        // stored sample (extent 1)
        if self.scalar || self.extent == 1 {
            info |= 0x400;
        }
        let mut out = info.to_le_bytes().to_vec();
        out.push(1); // number of samples
        out.push(self.name.len() as u8);
        out.extend_from_slice(self.name.as_bytes());
        out
    }

    /// The bytes AbcCoreOgawa `HashPropertyHeader` hashes: name, metadata, pod, extent (a 0 for
    /// a scalar), time per cycle 1.0, one stored time 0.0
    fn header_hash_bytes(&self) -> Vec<u8> {
        let mut d = self.name.as_bytes().to_vec();
        d.extend_from_slice(self.meta.as_bytes());
        d.push(self.pod);
        d.push(self.extent);
        if self.scalar {
            d.push(0);
        }
        d.extend_from_slice(&1.0f64.to_le_bytes());
        d.extend_from_slice(&1u32.to_le_bytes());
        d.extend_from_slice(&0.0f64.to_le_bytes());
        d
    }
}

fn spooky(bytes: &[u8]) -> [u8; 16] {
    let (a, b) = spooky_hash128(bytes, 0, 0);
    let mut out = [0; 16];
    out[..8].copy_from_slice(&a.to_le_bytes());
    out[8..].copy_from_slice(&b.to_le_bytes());
    out
}

/// Writes one property's sample and returns its group and its hash
fn write_prop(out: &mut Ogawa, prop: &Prop, raw: &[u8]) -> (u64, [u8; 16]) {
    let digest = murmur3_x64_128(raw);
    let mut hashed = prop.header_hash_bytes();
    let d = out.data(&[&digest, raw]);
    let group = if prop.scalar {
        hashed.extend_from_slice(&digest);
        out.group(&[d])
    } else {
        // rank 1: the dimensions are implied by the size, so they are an empty data child
        let mut dims = (prop.points as u64).to_le_bytes().to_vec();
        dims.extend_from_slice(&digest);
        hashed.extend_from_slice(&spooky(&dims));
        out.group(&[d, EMPTY_DATA])
    };
    (group, spooky(&hashed))
}

fn f32_bytes(v: impl Iterator<Item = f32>) -> Vec<u8> {
    v.flat_map(f32::to_le_bytes).collect()
}

fn i32_bytes(v: impl Iterator<Item = i32>) -> Vec<u8> {
    v.flat_map(i32::to_le_bytes).collect()
}

/// The archive bytes of `mesh`
fn archive_bytes(mesh: &Mesh, config: &AlembicConfig, provenance: &Provenance) -> Vec<u8> {
    let mut out = Ogawa::new();
    let mut map = MetaDataMap::default();
    let version = out.data(&[&OGAWA_FILE_VERSION.to_le_bytes()]);
    let library = out.data(&[&LIBRARY_VERSION.to_le_bytes()]);

    let n = mesh.vertices.len();
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for v in &mesh.vertices {
        // as Imath's Box::extendBy: a strict comparison in vertex order, so a tie between
        // 0.0 and -0.0 keeps the first one and the stored bounds do not depend on a min / max
        for (k, c) in v.position.to_array().into_iter().enumerate() {
            let c = f64::from(c);
            if c < lo[k] {
                lo[k] = c;
            }
            if c > hi[k] {
                hi[k] = c;
            }
        }
    }
    if n == 0 {
        lo = [0.0; 3];
        hi = [0.0; 3];
    }
    let bounds: Vec<u8> = lo.iter().chain(&hi).flat_map(|x| x.to_le_bytes()).collect();
    let positions = f32_bytes(mesh.vertices.iter().flat_map(|v| v.position.to_array()));
    // Alembic stores faces clockwise
    let indices = i32_bytes(
        mesh.indices
            .chunks_exact(3)
            .flat_map(|t| [t[0] as i32, t[2] as i32, t[1] as i32]),
    );
    let tris = mesh.indices.len() / 3;
    let counts = i32_bytes(std::iter::repeat_n(3, tris));

    let mut props = vec![
        (
            Prop {
                name: ".selfBnds",
                meta: MD_BOX,
                scalar: true,
                pod: POD_FLOAT64,
                extent: 6,
                points: 1,
            },
            bounds,
        ),
        (
            Prop {
                name: "P",
                meta: MD_POINT,
                scalar: false,
                pod: POD_FLOAT32,
                extent: 3,
                points: n,
            },
            positions,
        ),
        (
            Prop {
                name: ".faceIndices",
                meta: "",
                scalar: false,
                pod: POD_INT32,
                extent: 1,
                points: tris * 3,
            },
            indices,
        ),
        (
            Prop {
                name: ".faceCounts",
                meta: "",
                scalar: false,
                pod: POD_INT32,
                extent: 1,
                points: tris,
            },
            counts,
        ),
    ];
    if config.export_uvs {
        let uv = f32_bytes(mesh.vertices.iter().flat_map(|v| v.uv.to_array()));
        props.push((
            Prop {
                name: "uv",
                meta: MD_UV,
                scalar: false,
                pod: POD_FLOAT32,
                extent: 2,
                points: n,
            },
            uv,
        ));
    }
    if config.export_normals {
        let normals = f32_bytes(mesh.vertices.iter().flat_map(|v| v.normal.to_array()));
        props.push((
            Prop {
                name: "N",
                meta: MD_NORMAL,
                scalar: false,
                pod: POD_FLOAT32,
                extent: 3,
                points: n,
            },
            normals,
        ));
    }

    // the .geom compound: its properties, then their headers
    let mut children = Vec::new();
    let mut hashes = Vec::new();
    for (prop, raw) in &props {
        let (g, h) = write_prop(&mut out, prop, raw);
        children.push(g);
        hashes.extend_from_slice(&h);
    }
    let mut headers = Vec::new();
    for (prop, _) in &props {
        headers.extend(prop.info(&mut map));
    }
    let headers = out.data(&[&headers]);
    children.push(headers);
    let geom = out.group(&children);
    let geom_header = Prop {
        name: ".geom",
        meta: MD_GEOM,
        scalar: false,
        pod: 0,
        extent: 0,
        points: 0,
    };
    hashes.extend_from_slice(geom_header.name.as_bytes());
    hashes.extend_from_slice(geom_header.meta.as_bytes());
    let geom_hash = spooky(&hashes);

    // the object's properties: the .geom compound and its header (a compound: the index and name)
    let mut compound_info = (u32::from(map.index(MD_GEOM)) << 20).to_le_bytes().to_vec();
    compound_info.push(geom_header.name.len() as u8);
    compound_info.extend_from_slice(geom_header.name.as_bytes());
    let info = out.data(&[&compound_info]);
    let object_props = out.group(&[geom, info]);

    // the mesh object: its properties, then its hashes (properties, no children)
    let data_hash = spooky(&geom_hash);
    let mesh_hashes = out.data(&[&data_hash, &[0u8; 16]]);
    let mesh_group = out.group(&[object_props, mesh_hashes]);
    let mut object_hashed = data_hash.to_vec();
    object_hashed.extend_from_slice(MD_OBJECT.as_bytes());
    object_hashed.extend_from_slice(b"mesh");
    let object_hash = spooky(&object_hashed);

    // the top object: no properties, the mesh object, then the child header and its hashes
    let mut top_headers = 4u32.to_le_bytes().to_vec();
    top_headers.extend_from_slice(b"mesh");
    top_headers.push(map.index(MD_OBJECT));
    let top_hashes = [spooky(&[]), spooky(&object_hash)].concat();
    let top_data = out.data(&[&top_headers, &top_hashes]);
    let top = out.group(&[EMPTY_GROUP, mesh_group, top_data]);

    let meta = format!(
        "_ai_Application=ALICE-SDF {};_ai_DCC_FPS={};{KEY}={}",
        env!("CARGO_PKG_VERSION"),
        config.fps,
        provenance.hex()
    );
    let archive_meta = out.data(&[meta.as_bytes()]);
    // one time sampling (identity), its largest sample count 1
    let mut sampling = 1u32.to_le_bytes().to_vec();
    sampling.extend_from_slice(&1.0f64.to_le_bytes());
    sampling.extend_from_slice(&1u32.to_le_bytes());
    sampling.extend_from_slice(&0.0f64.to_le_bytes());
    let time_samplings = out.data(&[&sampling]);
    let indexed = out.data(&[&map.bytes()]);
    let root = out.group(&[version, library, top, archive_meta, time_samplings, indexed]);
    out.finish(root)
}

/// MurmurHash3 x64 128 with seed 0 (the 16-byte digest Alembic stores before each sample)
fn murmur3_x64_128(data: &[u8]) -> [u8; 16] {
    const C1: u64 = 0x87c3_7b91_1142_53d5;
    const C2: u64 = 0x4cf5_ad43_2745_937f;
    let fmix = |mut k: u64| {
        k ^= k >> 33;
        k = k.wrapping_mul(0xff51_afd7_ed55_8ccd);
        k ^= k >> 33;
        k = k.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
        k ^ (k >> 33)
    };
    let (mut h1, mut h2) = (0u64, 0u64);
    let mut blocks = data.chunks_exact(16);
    for b in &mut blocks {
        let k1 = u64::from_le_bytes(b[..8].try_into().unwrap_or([0; 8]));
        let k2 = u64::from_le_bytes(b[8..].try_into().unwrap_or([0; 8]));
        h1 ^= k1.wrapping_mul(C1).rotate_left(31).wrapping_mul(C2);
        h1 = h1
            .rotate_left(27)
            .wrapping_add(h2)
            .wrapping_mul(5)
            .wrapping_add(0x52dc_e729);
        h2 ^= k2.wrapping_mul(C2).rotate_left(33).wrapping_mul(C1);
        h2 = h2
            .rotate_left(31)
            .wrapping_add(h1)
            .wrapping_mul(5)
            .wrapping_add(0x3849_5ab5);
    }
    let tail = blocks.remainder();
    let mut t = [0u8; 16];
    t[..tail.len()].copy_from_slice(tail);
    if tail.len() > 8 {
        let k2 = u64::from_le_bytes(t[8..].try_into().unwrap_or([0; 8]));
        h2 ^= k2.wrapping_mul(C2).rotate_left(33).wrapping_mul(C1);
    }
    if !tail.is_empty() {
        let k1 = u64::from_le_bytes(t[..8].try_into().unwrap_or([0; 8]));
        h1 ^= k1.wrapping_mul(C1).rotate_left(31).wrapping_mul(C2);
    }
    let len = data.len() as u64;
    h1 ^= len;
    h2 ^= len;
    h1 = h1.wrapping_add(h2);
    h2 = h2.wrapping_add(h1);
    h1 = fmix(h1);
    h2 = fmix(h2);
    h1 = h1.wrapping_add(h2);
    h2 = h2.wrapping_add(h1);
    let mut out = [0; 16];
    out[..8].copy_from_slice(&h1.to_le_bytes());
    out[8..].copy_from_slice(&h2.to_le_bytes());
    out
}

const SC_CONST: u64 = 0xdead_beef_dead_beef;

/// SpookyHash V2 `Hash128` (Alembic's object and property hashes; an incremental `Update` /
/// `Final` over pieces equals this over their concatenation)
fn spooky_hash128(msg: &[u8], seed1: u64, seed2: u64) -> (u64, u64) {
    if msg.len() < 192 {
        return spooky_short(msg, seed1, seed2);
    }
    let word =
        |b: &[u8], i: usize| u64::from_le_bytes(b[i * 8..i * 8 + 8].try_into().unwrap_or([0; 8]));
    let mut h = [
        seed1, seed2, SC_CONST, seed1, seed2, SC_CONST, seed1, seed2, SC_CONST, seed1, seed2,
        SC_CONST,
    ];
    let mut blocks = msg.chunks_exact(96);
    for b in &mut blocks {
        let d: [u64; 12] = core::array::from_fn(|i| word(b, i));
        spooky_mix(&d, &mut h);
    }
    let rem = blocks.remainder();
    let mut last = [0u8; 96];
    last[..rem.len()].copy_from_slice(rem);
    last[95] = rem.len() as u8;
    let d: [u64; 12] = core::array::from_fn(|i| word(&last, i));
    for (hi, di) in h.iter_mut().zip(d) {
        *hi = hi.wrapping_add(di);
    }
    for _ in 0..3 {
        spooky_end_partial(&mut h);
    }
    (h[0], h[1])
}

fn spooky_mix(d: &[u64; 12], s: &mut [u64; 12]) {
    const ROT: [u32; 12] = [11, 32, 43, 31, 17, 28, 39, 57, 55, 54, 22, 46];
    for (i, (&di, &r)) in d.iter().zip(&ROT).enumerate() {
        s[i] = s[i].wrapping_add(di);
        s[(i + 2) % 12] ^= s[(i + 10) % 12];
        s[(i + 11) % 12] ^= s[i];
        s[i] = s[i].rotate_left(r);
        s[(i + 11) % 12] = s[(i + 11) % 12].wrapping_add(s[(i + 1) % 12]);
    }
}

fn spooky_end_partial(h: &mut [u64; 12]) {
    const ROT: [u32; 12] = [44, 15, 34, 21, 38, 33, 10, 13, 38, 53, 42, 54];
    for (i, &r) in ROT.iter().enumerate() {
        let (a, b, c) = ((i + 11) % 12, (i + 1) % 12, (i + 2) % 12);
        h[a] = h[a].wrapping_add(h[b]);
        h[c] ^= h[a];
        h[b] = h[b].rotate_left(r);
    }
}

fn spooky_short_mix(h: &mut [u64; 4]) {
    for [r0, r1, r2, r3] in [[50, 52, 30, 41], [54, 48, 38, 37], [62, 34, 5, 36]] {
        h[2] = h[2].rotate_left(r0).wrapping_add(h[3]);
        h[0] ^= h[2];
        h[3] = h[3].rotate_left(r1).wrapping_add(h[0]);
        h[1] ^= h[3];
        h[0] = h[0].rotate_left(r2).wrapping_add(h[1]);
        h[2] ^= h[0];
        h[1] = h[1].rotate_left(r3).wrapping_add(h[2]);
        h[3] ^= h[1];
    }
}

fn spooky_short_end(h: &mut [u64; 4]) {
    for (x, y, r) in [
        (3, 2, 15),
        (0, 3, 52),
        (1, 0, 26),
        (2, 1, 51),
        (3, 2, 28),
        (0, 3, 9),
        (1, 0, 47),
        (2, 1, 54),
        (3, 2, 32),
        (0, 3, 25),
        (1, 0, 63),
    ] {
        h[x] ^= h[y];
        h[y] = h[y].rotate_left(r);
        h[x] = h[x].wrapping_add(h[y]);
    }
}

fn spooky_short(msg: &[u8], seed1: u64, seed2: u64) -> (u64, u64) {
    let le = |b: &[u8]| {
        let mut w = [0u8; 8];
        w[..b.len()].copy_from_slice(b);
        u64::from_le_bytes(w)
    };
    let mut h = [seed1, seed2, SC_CONST, SC_CONST];
    let mut p = 0;
    let mut rem = msg.len() % 32;
    if msg.len() > 15 {
        for _ in 0..msg.len() / 32 {
            h[2] = h[2].wrapping_add(le(&msg[p..p + 8]));
            h[3] = h[3].wrapping_add(le(&msg[p + 8..p + 16]));
            spooky_short_mix(&mut h);
            h[0] = h[0].wrapping_add(le(&msg[p + 16..p + 24]));
            h[1] = h[1].wrapping_add(le(&msg[p + 24..p + 32]));
            p += 32;
        }
        if rem >= 16 {
            h[2] = h[2].wrapping_add(le(&msg[p..p + 8]));
            h[3] = h[3].wrapping_add(le(&msg[p + 8..p + 16]));
            spooky_short_mix(&mut h);
            p += 16;
            rem -= 16;
        }
    }
    h[3] = h[3].wrapping_add((msg.len() as u64) << 56);
    let t = &msg[p..p + rem];
    if rem >= 8 {
        h[2] = h[2].wrapping_add(le(&t[..8]));
        h[3] = h[3].wrapping_add(le(&t[8..]));
    } else if rem > 0 {
        h[2] = h[2].wrapping_add(le(t));
    } else {
        h[2] = h[2].wrapping_add(SC_CONST);
        h[3] = h[3].wrapping_add(SC_CONST);
    }
    spooky_short_end(&mut h);
    (h[0], h[1])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// oracle: the digests the Alembic library 1.8.12 stored for these samples (a library-written
    /// archive of one triangle, read back with its reader)
    #[test]
    fn murmur3_matches_the_alembic_library() {
        let p: Vec<u8> = [0f32, 0., 0., 1., 0., 0., 0., 1., 0.]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let hex = |d: [u8; 16]| {
            d.iter().fold(String::new(), |mut s, b| {
                use std::fmt::Write as _;
                let _ = write!(s, "{b:02x}");
                s
            })
        };
        assert_eq!(hex(murmur3_x64_128(&p)), "740a45f98aeef44cba0a3ca6c0b13225");
        let fi: Vec<u8> = [0i32, 1, 2].iter().flat_map(|i| i.to_le_bytes()).collect();
        assert_eq!(
            hex(murmur3_x64_128(&fi)),
            "83bc3ec3f049daf0c0d9f0a4b3b3e26d"
        );
        assert_eq!(
            hex(murmur3_x64_128(&3i32.to_le_bytes())),
            "3ba1d27b7fde4848a86e865c6c402dfa"
        );
    }

    /// oracle: Alembic::Util::SpookyHash of the library 1.8.12 (Init(0, 0), Update in two pieces,
    /// Final) over bytes `(i * 37 + 11) & 255`, across the short / long and remainder branches
    #[test]
    fn spooky_matches_the_alembic_library() {
        let cases: [(usize, u64, u64); 9] = [
            (0, 0x2327_06fc_6bf5_0919, 0x8b72_ee65_b4e8_51c7),
            (7, 0xc9d3_e5dd_3590_09d7, 0x9696_d9e8_bdfa_7e8c),
            (12, 0x31a8_b16c_9478_7304, 0x957a_da37_24f0_9804),
            (15, 0x0b27_cadc_e791_1a15, 0xafd8_fa48_8ce9_8414),
            (47, 0xca0f_bc41_9afe_ee0f, 0xdd27_698d_0d1a_0d09),
            (191, 0x2a6d_3a57_5ee1_082a, 0x573f_d589_12f9_b7f5),
            (192, 0x0907_e64f_76e4_62fd, 0xda62_f180_36cb_a74b),
            (193, 0x69f2_af75_7102_e549, 0x8d49_9e7a_7dcd_5479),
            (500, 0xe41f_9a12_beb9_223b, 0x8aa0_e15d_ec9f_bf0c),
        ];
        for (n, a, b) in cases {
            let v: Vec<u8> = (0..n).map(|i| ((i * 37 + 11) & 255) as u8).collect();
            assert_eq!(spooky_hash128(&v, 0, 0), (a, b), "length {n}");
        }
    }

    #[test]
    fn the_archive_starts_with_a_finished_ogawa_header() {
        let mesh = crate::mesh::sdf_to_mesh(
            &crate::types::SdfNode::sphere(1.0),
            glam::Vec3::splat(-2.0),
            glam::Vec3::splat(2.0),
            &crate::mesh::MarchingCubesConfig {
                resolution: 8,
                ..Default::default()
            },
        );
        let b = archive_bytes(&mesh, &AlembicConfig::default(), &Provenance::sdf());
        assert_eq!(&b[..8], b"Ogawa\xFF\x00\x01");
        let root = u64::from_le_bytes(b[8..16].try_into().unwrap()) as usize;
        assert_eq!(u64::from_le_bytes(b[root..root + 8].try_into().unwrap()), 6);
    }
}

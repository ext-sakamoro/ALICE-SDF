//! Kink cells: the exact boundary channel `d_b` where trilinear interpolation cannot follow it.
//!
//! `d_b` (module docs of [`super`]) bends at the rim's tips and corners: it is the distance to the rim, a 1-Lipschitz
//! function with creases, and a trilinear cell cannot follow a crease inside it (the error is bounded only by about
//! `0.87 h`). On thin strands that error reads a point of the sheet as beyond the rim (`d_b > step/2`), so the march
//! rejects a crossing that is there. The cells the rim crosses, and their 26 neighbours on the same level (the
//! kink cells), therefore carry references to the sheet's own triangles and rim segments, and there `d_b` is computed
//! from them exactly instead of interpolated.
//!
//! Definition (the builder's): `q` the closest point of the sheet to `p`, `r = p - q`.
//! - The closest feature is a rim edge (its one face) or a rim vertex: `d_b = sqrt(|r|^2 - (r.n)^2)`, `n` that face's
//!   normal or the vertex pseudonormal (normalized).
//! - Otherwise: `d_b = -max(0, min(dist_to_rim(q), H - |r|))`, `H` the slab, `dist_to_rim` over the cell's rim
//!   segments (`1e3` mm when it lists none).
//!
//! The reference lists are built so that they provably hold every triangle and rim segment the exact value of any point
//! of the cell uses (cell centre `c`, diagonal `D`, `d_c` the distance from `c` to the sheet, `rho` from `c` to the
//! rim: triangles within `R1 = d_c + D`, segments within `R1 + min(rho + R1, H)`).
//!
//! On disk (the ASKC sidecar, little-endian; the ASHF file is unchanged): `b"ASKC"`, `u32` version, from v4 the
//! 32-byte semantics identifier the sidecar was written under ([`crate::provenance`]), `u32` sheets
//! (= the ASHF file's sheets, same order); per sheet: `f32` H, `f64` origin[3], `u32` nv, `f32` V[nv*3] (relative to
//! the origin), `u32` nf, per face `u32` a, b, c, `f32` n[3], `u32` rim mask (bit 0 edge ab, 1 bc, 2 ca); `u32` rim
//! vertices, per one `u32` v, `f32` pseudonormal[3]; `u32` segments, per one `u32` a, b; `u32` levels (= the sheet's
//! levels), per level `u32` kink bricks, per brick `u32` brick (its active number), `u32` mask[16] (bit `c` = cell
//! `c = x*64 + y*8 + z`), `u32` first (its first cell record); `u32` cell records, per one `u32` off, `u16` triangles,
//! `u16` segments; `u32` refs, `u32` ref[refs] (per cell its triangles, then its segments).

use std::io::{self, Read};
use std::sync::Arc;

use glam::{DVec3, Vec3};

/// The 4 bytes an ASKC file starts with.
pub const ASKC_MAGIC: [u8; 4] = *b"ASKC";
/// The ASKC version this reader takes (v3 with the semantics identifier after the version).
pub const ASKC_VERSION: u32 = 4;
/// The oldest ASKC version this reader still takes: versions 1 to 3 carry no semantics identifier and are refused.
pub const ASKC_VERSION_MIN: u32 = 4;
/// The march's step margin for a field with kink cells.
///
/// Every step is taken from the union value less this (the s cover's position bound `eps`,
/// [`super::kink_build::kink_sheet`]), so a step whose start is farther than the margin from every crossing the field
/// can show stays free of them whatever the rounding of `t` (CPU and GPU alike).
pub const KINK_STEP_MARGIN: f32 = 0.01;
/// Cell flag: the march computes `d_b` exactly in this cell.
pub const CELL_DB: u32 = 1;
/// Cell flag: the march computes `s` (and the unsigned distance) exactly in this cell.
pub const CELL_S: u32 = 2;
/// Cell flag: the march takes the UV / author normal of the exact closest point in this cell (around a seam).
///
/// Every flagged cell does, when the file carries the attributes; this flag marks cells flagged for that alone.
pub const CELL_ATTR: u32 = 4;
/// `dist_to_rim` when a cell lists no rim segment (the builder's value).
const NO_RIM: f64 = 1e3;

/// Why an ASKC file was refused.
#[derive(Debug)]
pub enum KinkError {
    /// Reading failed.
    Io(io::Error),
    /// The file is not a valid ASKC sidecar for the field it is attached to.
    Format(String),
    /// The sidecar was written under another semantics identifier than this crate's (or one it still reproduces).
    Semantics(crate::io::IoError),
}

impl std::fmt::Display for KinkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "ASKC i/o: {e}"),
            Self::Format(s) => write!(f, "not a valid ASKC sidecar: {s}"),
            Self::Semantics(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for KinkError {}

impl From<io::Error> for KinkError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

#[derive(Debug, Clone, Copy)]
struct Face {
    v: [u32; 3],
    n: DVec3,
    rim: u8,
    /// Edge pseudonormals by the opposite corner (0: edge bc, 1: ca, 2: ab); zero in a v1 file.
    epn: [DVec3; 3],
}

#[derive(Debug, Clone, Copy)]
struct Cell {
    off: u32,
    ntri: u16,
    nseg: u16,
    flags: u32,
}

/// The UV and author normal of a point (each when the sheet carries it).
pub type Attributes = (Option<[f32; 2]>, Option<[f32; 3]>);

/// The exact values at a point of a flagged cell ([`CELL_DB`], [`CELL_S`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Exact {
    /// `s` when the cell is an s cell.
    pub s: Option<f32>,
    /// `d_b` when the cell is a d_b cell.
    pub db: Option<f32>,
    /// The unsigned distance to the sheet (from the cell's triangles).
    pub ud: f32,
    /// The UV of the closest point (its face's corner UVs by the barycentric weights), when the file carries UVs.
    pub uv: Option<[f32; 2]>,
    /// The author normal of the closest point (the corners' normals by the weights, not normalized), when carried.
    pub nrm: Option<[f32; 3]>,
}

#[derive(Debug, Clone, Default)]
struct KinkLevel {
    /// Per active brick: its slot in `masks` / `first`, or `u32::MAX` (no kink cell).
    slot: Vec<u32>,
    masks: Vec<[u32; 16]>,
    first: Vec<u32>,
}

/// The kink cells of one sheet and the mesh data their exact `d_b` reads.
#[derive(Debug, Clone)]
pub struct KinkSheet {
    slab: f64,
    origin: DVec3,
    verts: Vec<DVec3>,
    faces: Vec<Face>,
    /// Per vertex: the normalized pseudonormal when it is a rim vertex.
    rim_vn: Vec<Option<DVec3>>,
    /// Per vertex: the pseudonormal as the builder holds it (the sign of `s` at a vertex); v1: rim vertices only.
    vn: Vec<DVec3>,
    /// Per face: the corner UVs (empty when the file carries none).
    face_uv: Vec<[[f32; 2]; 3]>,
    /// Per face: the corner author normals (empty when the file carries none).
    face_nrm: Vec<[[f32; 3]; 3]>,
    segs: Vec<[u32; 2]>,
    levels: Vec<KinkLevel>,
    cells: Vec<Cell>,
    refs: Vec<u32>,
}

/// The closest feature of a triangle to a point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Feature {
    Face,
    /// Edge 0 = ab, 1 = bc, 2 = ca.
    Edge(u8),
    /// Corner 0 = a, 1 = b, 2 = c.
    Vertex(u8),
}

/// Closest point of triangle `abc` to `p` and its feature (Ericson, Real-Time Collision Detection 5.1.5). A degenerate
/// triangle is answered by its three edges.
fn closest_on_tri(p: DVec3, a: DVec3, b: DVec3, c: DVec3) -> (DVec3, Feature) {
    let ab = b - a;
    let ac = c - a;
    if ab.cross(ac).length_squared() <= 0.0 {
        let mut best = (a, Feature::Vertex(0), f64::INFINITY);
        for (i, (x, y)) in [(a, b), (b, c), (c, a)].into_iter().enumerate() {
            let e = y - x;
            let ee = e.length_squared();
            let t = if ee > 0.0 {
                ((p - x).dot(e) / ee).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let q = x + e * t;
            let d = (p - q).length_squared();
            if d < best.2 {
                let f = if t <= 0.0 {
                    Feature::Vertex(i as u8)
                } else if t >= 1.0 {
                    Feature::Vertex(((i + 1) % 3) as u8)
                } else {
                    Feature::Edge(i as u8)
                };
                best = (q, f, d);
            }
        }
        return (best.0, best.1);
    }
    let ap = p - a;
    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return (a, Feature::Vertex(0));
    }
    let bp = p - b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return (b, Feature::Vertex(1));
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        return (a + ab * (d1 / (d1 - d3)), Feature::Edge(0));
    }
    let cp = p - c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        return (c, Feature::Vertex(2));
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        return (a + ac * (d2 / (d2 - d6)), Feature::Edge(2));
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        return (
            b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6))),
            Feature::Edge(1),
        );
    }
    let denom = 1.0 / (va + vb + vc);
    (a + ab * (vb * denom) + ac * (vc * denom), Feature::Face)
}

fn seg_dist(q: DVec3, a: DVec3, b: DVec3) -> f64 {
    let e = b - a;
    let ee = e.length_squared();
    let t = if ee > 0.0 {
        ((q - a).dot(e) / ee).clamp(0.0, 1.0)
    } else {
        0.0
    };
    (q - (a + e * t)).length()
}

/// Barycentric weights of `q` (on the plane of `abc`) in triangle `abc`: the builder's formula (`v0 = b - a`,
/// `v1 = c - a`, `v2 = q - a`); a degenerate triangle gives the weights of its first corner.
#[allow(
    clippy::suspicious_operation_groupings,
    reason = "the 2x2 determinant d00 d11 - d01^2 of the barycentric solve"
)]
fn bary(q: DVec3, a: DVec3, b: DVec3, c: DVec3) -> [f64; 3] {
    let (v0, v1, v2) = (b - a, c - a, q - a);
    let (d00, d01, d11, d20, d21) = (v0.dot(v0), v0.dot(v1), v1.dot(v1), v2.dot(v0), v2.dot(v1));
    let den = d00 * d11 - d01 * d01;
    if den == 0.0 {
        return [1.0, 0.0, 0.0];
    }
    let v = (d11 * d20 - d01 * d21) / den;
    let w = (d00 * d21 - d01 * d20) / den;
    [1.0 - v - w, v, w]
}

impl KinkSheet {
    /// The kink cell record of (level, active brick, cell) when that cell is a kink cell.
    fn cell(&self, level: usize, brick: usize, cell: usize) -> Option<Cell> {
        let lv = self.levels.get(level)?;
        let slot = *lv.slot.get(brick)?;
        if slot == u32::MAX {
            return None;
        }
        let m = &lv.masks[slot as usize];
        let (w, bit) = (cell / 32, cell % 32);
        if m[w] >> bit & 1 == 0 {
            return None;
        }
        let below: u32 = m[..w].iter().map(|x| x.count_ones()).sum::<u32>()
            + (m[w] & ((1u32 << bit) - 1)).count_ones();
        Some(self.cells[(lv.first[slot as usize] + below) as usize])
    }

    /// The exact values at `p` when `p`'s cell is a flagged cell: `s` in an s cell, `d_b` in a `d_b` cell, and the
    /// unsigned distance from the cell's triangles.
    #[must_use]
    pub fn exact_at(&self, level: usize, brick: usize, cell: usize, p: Vec3) -> Option<Exact> {
        let c = self.cell(level, brick, cell)?;
        let off = c.off as usize;
        let tris = &self.refs[off..off + c.ntri as usize];
        let segs = &self.refs[off + c.ntri as usize..off + c.ntri as usize + c.nseg as usize];
        let pl = p.as_dvec3() - self.origin;
        let Some(nf) = self.nearest(pl, tris) else {
            // no triangle listed (not produced by the builder): no surface within the cell's radius
            return Some(Exact {
                s: (c.flags & CELL_S != 0).then_some(NO_RIM as f32),
                db: (c.flags & CELL_DB != 0).then_some(NO_RIM as f32),
                ud: NO_RIM as f32,
                uv: None,
                nrm: None,
            });
        };
        let (uv, nrm) = self.attributes(nf);
        Some(Exact {
            uv,
            nrm,
            s: (c.flags & CELL_S != 0).then(|| self.exact_s(pl, nf) as f32),
            db: (c.flags & CELL_DB != 0).then(|| self.exact_db(pl, nf, segs) as f32),
            ud: (pl - nf.1).length() as f32,
        })
    }

    /// The UV and author normal at the closest point `q` of face `f` (the builder's: the corners' values by the
    /// barycentric weights of `q`), each when the file carries it.
    fn attributes(
        &self,
        (f, q, _): (usize, DVec3, Feature),
    ) -> (Option<[f32; 2]>, Option<[f32; 3]>) {
        if self.face_uv.is_empty() && self.face_nrm.is_empty() {
            return (None, None);
        }
        let [a, b, c] = self.faces[f].v.map(|i| self.verts[i as usize]);
        let w = bary(q, a, b, c);
        let uv = self.face_uv.get(f).map(|t| {
            core::array::from_fn(|k| (0..3).map(|i| w[i] * f64::from(t[i][k])).sum::<f64>() as f32)
        });
        let nrm = self.face_nrm.get(f).map(|t| {
            core::array::from_fn(|k| (0..3).map(|i| w[i] * f64::from(t[i][k])).sum::<f64>() as f32)
        });
        (uv, nrm)
    }

    /// The flags of (level, active brick, cell) when it is a flagged cell.
    #[must_use]
    pub fn flags(&self, level: usize, brick: usize, cell: usize) -> Option<u32> {
        self.cell(level, brick, cell).map(|c| c.flags)
    }

    /// The nearest of `tris` to local `p`: face, closest point, feature (the first of equal distances wins: the
    /// builder's order, strictly nearer by more than 1e-15).
    fn nearest(&self, p: DVec3, tris: &[u32]) -> Option<(usize, DVec3, Feature)> {
        let mut best = f64::INFINITY;
        let mut hit = None;
        for &f in tris {
            let fc = &self.faces[f as usize];
            let [a, b, c] = fc.v.map(|i| self.verts[i as usize]);
            let (q, feat) = closest_on_tri(p, a, b, c);
            let d = (p - q).length();
            if d < best - 1e-15 {
                best = d;
                hit = Some((f as usize, q, feat));
            }
        }
        hit
    }

    /// The boundary normal when the closest feature is a rim edge (its face's normal) or a rim vertex (normalized).
    fn boundary_normal(&self, f: usize, feat: Feature) -> Option<DVec3> {
        let fc = &self.faces[f];
        match feat {
            Feature::Edge(e) if fc.rim >> e & 1 == 1 => Some(fc.n),
            Feature::Vertex(i) => self.rim_vn[fc.v[i as usize] as usize],
            _ => None,
        }
    }

    /// The exact `s` at local `p` (module docs): `r.n` on a rim feature, else the distance signed by the closest
    /// feature's pseudonormal (face normal, edge pseudonormal, vertex pseudonormal).
    fn exact_s(&self, p: DVec3, (f, q, feat): (usize, DVec3, Feature)) -> f64 {
        let r = p - q;
        if let Some(n) = self.boundary_normal(f, feat) {
            return r.dot(n);
        }
        let ud = r.length();
        if ud < 1e-12 {
            return 0.0;
        }
        let fc = &self.faces[f];
        let n = match feat {
            Feature::Face => fc.n,
            // edge e (ab, bc, ca) is opposite corner (e + 2) % 3
            Feature::Edge(e) => fc.epn[(usize::from(e) + 2) % 3],
            Feature::Vertex(i) => self.vn[fc.v[usize::from(i)] as usize],
        };
        let sn = r.dot(n);
        if sn > 0.0 {
            ud
        } else if sn < 0.0 {
            -ud
        } else {
            0.0
        }
    }

    /// The exact `d_b` at local `p` from the nearest triangle and the given rim segments (module docs).
    fn exact_db(&self, p: DVec3, (f, q, feat): (usize, DVec3, Feature), segs: &[u32]) -> f64 {
        let r = p - q;
        if let Some(n) = self.boundary_normal(f, feat) {
            let sn = r.dot(n);
            return (r.length_squared() - sn * sn).max(0.0).sqrt();
        }
        let dr = segs.iter().fold(NO_RIM, |m, &s| {
            let [a, b] = self.segs[s as usize];
            m.min(seg_dist(q, self.verts[a as usize], self.verts[b as usize]))
        });
        -(dr.min(self.slab - r.length())).max(0.0)
    }
}

impl KinkSheet {
    // ALLOW-UNWIRED: wiring debt kink-point-evaluator, oracle src/sheet_field/kink_build.rs::derived_normals_by_hand_and_the_point_evaluator
    /// The sheet's mesh with the shared normals ([`super::kink_build::derived_normals`]) and no cells: the point
    /// evaluator a builder uses for its node values, the same code the march runs in a flagged cell.
    #[must_use]
    pub fn from_geom(geom: &super::kink_build::SheetGeom) -> Self {
        let (fnrm, epn, vn) = super::kink_build::derived_normals(geom);
        let rim: std::collections::BTreeSet<(u32, u32)> = geom
            .rim
            .iter()
            .map(|e| (e[0].min(e[1]), e[0].max(e[1])))
            .collect();
        let k = |a: u32, b: u32| rim.contains(&(a.min(b), a.max(b)));
        let mut rim_vn = vec![None; geom.verts.len()];
        for e in geom.rim {
            for &v in e {
                let n = vn[v as usize];
                rim_vn[v as usize] = Some(n / n.length().max(1e-12));
            }
        }
        Self {
            slab: geom.slab,
            origin: DVec3::ZERO,
            verts: geom.verts.to_vec(),
            faces: geom
                .faces
                .iter()
                .zip(&fnrm)
                .zip(&epn)
                .map(|((t, n), e)| Face {
                    v: *t,
                    n: *n,
                    rim: u8::from(k(t[0], t[1]))
                        | (u8::from(k(t[1], t[2])) << 1)
                        | (u8::from(k(t[2], t[0])) << 2),
                    epn: *e,
                })
                .collect(),
            rim_vn,
            vn,
            face_uv: geom
                .uv
                .map(|u| u.iter().map(|t| t.map(|c| c.map(|x| x as f32))).collect())
                .unwrap_or_default(),
            face_nrm: geom
                .nrm
                .map(|u| u.iter().map(|t| t.map(|c| c.map(|x| x as f32))).collect())
                .unwrap_or_default(),
            segs: geom.rim.to_vec(),
            levels: Vec::new(),
            cells: Vec::new(),
            refs: Vec::new(),
        }
    }

    // ALLOW-UNWIRED: wiring debt kink-point-evaluator, oracle src/sheet_field/kink_build.rs::derived_normals_by_hand_and_the_point_evaluator
    /// `(s, d_b, unsigned distance)` at world point `p` from the candidate triangles `tris` (must hold the nearest
    /// face) and rim segments `segs` (must hold every rim segment within the slab of the closest point); `None` when
    /// `tris` is empty.
    #[must_use]
    pub fn eval_all(&self, p: DVec3, tris: &[u32], segs: &[u32]) -> Option<(f64, f64, f64)> {
        let pl = p - self.origin;
        let nf = self.nearest(pl, tris)?;
        Some((
            self.exact_s(pl, nf),
            self.exact_db(pl, nf, segs),
            (pl - nf.1).length(),
        ))
    }

    // ALLOW-UNWIRED: wiring debt kink-point-evaluator, oracle tests/test_sheet_field_kink.rs (attribute cells)
    /// The UV and author normal at world point `p`'s closest point among `tris` (the builder's attribute definition,
    /// for node values), each when carried; `None` when `tris` is empty.
    #[must_use]
    pub fn eval_attributes(&self, p: DVec3, tris: &[u32]) -> Option<Attributes> {
        let nf = self.nearest(p - self.origin, tris)?;
        Some(self.attributes(nf))
    }

    /// Number of faces (candidate indices run below it).
    #[must_use]
    pub const fn face_count(&self) -> usize {
        self.faces.len()
    }

    /// Number of rim segments (candidate indices run below it).
    #[must_use]
    pub const fn segment_count(&self) -> usize {
        self.segs.len()
    }
}

#[cfg(feature = "gpu")]
/// Layout of a sheet's kink block in the GPU index table (all `u32`; offsets relative to the block's first word).
#[cfg(feature = "gpu")]
pub(crate) mod gpu_layout {
    /// Words before the per-level table: slab, origin x y z, the offsets of verts, faces, segs, cells, refs, the level
    /// count, then the offsets of the per-face corner UVs (6 words) and author normals (9 words), `u32::MAX` if absent.
    pub const HEAD: usize = 12;
    /// Words per vertex: x, y, z (local, f32 bits), pseudonormal x, y, z (f32 bits, as the builder holds it), rim flag.
    pub const VERT: usize = 7;
    /// Words per face: a, b, c, normal x, y, z (f32 bits), rim mask, edge pseudonormals by the opposite corner (9).
    pub const FACE: usize = 16;
    /// Words per cell record: ref offset, triangles | segments << 16, flags.
    pub const CELL: usize = 3;
    /// Words per kink brick: the cell mask (16), the first cell record.
    pub const BRICK: usize = 17;
}

impl KinkSheet {
    /// The most triangles and the most segments one kink cell of the sheet refers to.
    #[must_use]
    pub fn max_cell_refs(&self) -> (u32, u32) {
        self.cells.iter().fold((0, 0), |(t, s), c| {
            (t.max(u32::from(c.ntri)), s.max(u32::from(c.nseg)))
        })
    }
}

#[cfg(feature = "gpu")]
impl KinkSheet {
    /// The sheet's kink block for the GPU index table ([`gpu_layout`]).
    pub(crate) fn gpu_block(&self) -> Vec<u32> {
        use gpu_layout::{BRICK, CELL, FACE, HEAD, VERT};
        let f = |x: f64| (x as f32).to_bits();
        let nl = self.levels.len();
        let mut b = vec![0u32; HEAD + 2 * nl];
        b[0] = f(self.slab);
        // the origin is a whole number of mm (the exporter floors it), exact in f32
        b[1] = f(self.origin.x);
        b[2] = f(self.origin.y);
        b[3] = f(self.origin.z);
        b[4] = b.len() as u32;
        for ((v, vn), rim) in self.verts.iter().zip(&self.vn).zip(&self.rim_vn) {
            b.extend([
                f(v.x),
                f(v.y),
                f(v.z),
                f(vn.x),
                f(vn.y),
                f(vn.z),
                u32::from(rim.is_some()),
            ]);
        }
        debug_assert_eq!(b.len(), b[4] as usize + VERT * self.verts.len());
        b[5] = b.len() as u32;
        for fc in &self.faces {
            b.extend(fc.v);
            b.extend([f(fc.n.x), f(fc.n.y), f(fc.n.z)]);
            b.push(u32::from(fc.rim));
            for e in &fc.epn {
                b.extend([f(e.x), f(e.y), f(e.z)]);
            }
        }
        debug_assert_eq!(b.len(), b[5] as usize + FACE * self.faces.len());
        b[6] = b.len() as u32;
        for sg in &self.segs {
            b.extend(sg);
        }
        b[7] = b.len() as u32;
        for c in &self.cells {
            b.extend([
                c.off,
                u32::from(c.ntri) | (u32::from(c.nseg) << 16),
                c.flags,
            ]);
        }
        debug_assert_eq!(b.len(), b[7] as usize + CELL * self.cells.len());
        b[8] = b.len() as u32;
        b.extend_from_slice(&self.refs);
        b[9] = nl as u32;
        b[10] = u32::MAX;
        b[11] = u32::MAX;
        if !self.face_uv.is_empty() {
            b[10] = b.len() as u32;
            for t in &self.face_uv {
                b.extend(t.iter().flatten().map(|x| x.to_bits()));
            }
        }
        if !self.face_nrm.is_empty() {
            b[11] = b.len() as u32;
            for t in &self.face_nrm {
                b.extend(t.iter().flatten().map(|x| x.to_bits()));
            }
        }
        for (li, lv) in self.levels.iter().enumerate() {
            b[HEAD + 2 * li] = b.len() as u32;
            b[HEAD + 2 * li + 1] = lv.slot.len() as u32;
            b.extend_from_slice(&lv.slot);
            for (m, first) in lv.masks.iter().zip(&lv.first) {
                b.extend_from_slice(m);
                b.push(*first);
            }
            debug_assert_eq!(
                b.len(),
                b[HEAD + 2 * li] as usize + lv.slot.len() + BRICK * lv.masks.len()
            );
        }
        b
    }
}

/// The kink cells of every sheet of one ASHF file (sheets in file order; `None` where a sheet has no rim).
#[derive(Debug, Clone, Default)]
pub struct KinkFile {
    sheets: Vec<Option<Arc<KinkSheet>>>,
}

struct Rd<'a, R: Read>(&'a mut R);

impl<R: Read> Rd<'_, R> {
    fn u32(&mut self) -> io::Result<u32> {
        let mut b = [0u8; 4];
        self.0.read_exact(&mut b)?;
        Ok(u32::from_le_bytes(b))
    }
    fn u16(&mut self) -> io::Result<u16> {
        let mut b = [0u8; 2];
        self.0.read_exact(&mut b)?;
        Ok(u16::from_le_bytes(b))
    }
    fn f32(&mut self) -> io::Result<f32> {
        Ok(f32::from_bits(self.u32()?))
    }
    fn f64(&mut self) -> io::Result<f64> {
        let mut b = [0u8; 8];
        self.0.read_exact(&mut b)?;
        Ok(f64::from_le_bytes(b))
    }
    fn v3(&mut self) -> io::Result<DVec3> {
        Ok(DVec3::new(
            f64::from(self.f32()?),
            f64::from(self.f32()?),
            f64::from(self.f32()?),
        ))
    }
    /// A count read from the file, refused above `cap` (a corrupt count must not allocate without bound).
    fn count(&mut self, what: &str, cap: usize) -> Result<usize, KinkError> {
        let n = self.u32()? as usize;
        if n > cap {
            return Err(KinkError::Format(format!("{what} {n} exceeds {cap}")));
        }
        Ok(n)
    }
}

const COUNT_CAP: usize = 1 << 28;

impl KinkFile {
    /// Read an ASKC sidecar. Every index is checked against the counts it refers to.
    pub fn read<R: Read>(r: &mut R) -> Result<Self, KinkError> {
        let mut rd = Rd(r);
        let mut magic = [0u8; 4];
        rd.0.read_exact(&mut magic)?;
        if magic != ASKC_MAGIC {
            return Err(KinkError::Format(format!("magic {magic:?}")));
        }
        let version = rd.u32()?;
        if !(ASKC_VERSION_MIN..=ASKC_VERSION).contains(&version) {
            return Err(KinkError::Format(format!(
                "version {version}, this reader takes {ASKC_VERSION_MIN}..={ASKC_VERSION}"
            )));
        }
        let mut id = [0u8; 32];
        rd.0.read_exact(&mut id)?;
        crate::provenance::sdf_load_verdict(Some(id), "ASKC", false)
            .map_err(KinkError::Semantics)?;
        let v2 = version >= 2;
        let ns = rd.count("sheets", COUNT_CAP)?;
        let mut sheets = Vec::with_capacity(ns);
        for k in 0..ns {
            let bad = |s: String| KinkError::Format(format!("sheet {k}: {s}"));
            let slab = f64::from(rd.f32()?);
            let origin = DVec3::new(rd.f64()?, rd.f64()?, rd.f64()?);
            // v3: which per-face attributes follow each face record (bit 0 corner UVs, bit 1 corner author normals)
            let attr = if version >= 3 { rd.u32()? } else { 0 };
            if attr > 3 {
                return Err(bad(format!("attribute flags {attr}")));
            }
            let nv = rd.count("vertices", COUNT_CAP)?;
            let mut verts = Vec::with_capacity(nv);
            let mut vn = vec![DVec3::ZERO; nv];
            let mut rim_vn = vec![None; nv];
            for i in 0..nv {
                verts.push(rd.v3()?);
                if v2 {
                    let n = rd.v3()?;
                    let fl = rd.u32()?;
                    vn[i] = n;
                    if fl & 1 == 1 {
                        rim_vn[i] = Some(n / n.length().max(1e-12));
                    }
                }
            }
            let nf = rd.count("faces", COUNT_CAP)?;
            let mut faces = Vec::with_capacity(nf);
            let mut face_uv = Vec::new();
            let mut face_nrm = Vec::new();
            for _ in 0..nf {
                let v = [rd.u32()?, rd.u32()?, rd.u32()?];
                let n = rd.v3()?;
                let rim = rd.u32()?;
                let epn = if v2 {
                    [rd.v3()?, rd.v3()?, rd.v3()?]
                } else {
                    [DVec3::ZERO; 3]
                };
                if attr & 1 == 1 {
                    let mut t = [[0f32; 2]; 3];
                    for c in &mut t {
                        *c = [rd.f32()?, rd.f32()?];
                    }
                    face_uv.push(t);
                }
                if attr & 2 == 2 {
                    let mut t = [[0f32; 3]; 3];
                    for c in &mut t {
                        *c = [rd.f32()?, rd.f32()?, rd.f32()?];
                    }
                    face_nrm.push(t);
                }
                if v.iter().any(|&i| i as usize >= nv) || rim > 7 {
                    return Err(bad(format!("face {v:?} rim {rim} (vertices {nv})")));
                }
                faces.push(Face {
                    v,
                    n,
                    rim: rim as u8,
                    epn,
                });
            }
            if !v2 {
                let nrv = rd.count("rim vertices", COUNT_CAP)?;
                for _ in 0..nrv {
                    let v = rd.u32()? as usize;
                    let n = rd.v3()?;
                    if v >= nv {
                        return Err(bad(format!("rim vertex {v} (vertices {nv})")));
                    }
                    vn[v] = n;
                    rim_vn[v] = Some(n / n.length().max(1e-12));
                }
            }
            let nseg = rd.count("segments", COUNT_CAP)?;
            let mut segs = Vec::with_capacity(nseg);
            for _ in 0..nseg {
                let s = [rd.u32()?, rd.u32()?];
                if s.iter().any(|&i| i as usize >= nv) {
                    return Err(bad(format!("segment {s:?} (vertices {nv})")));
                }
                segs.push(s);
            }
            let nl = rd.count("levels", 64)?;
            let mut levels = Vec::with_capacity(nl);
            let mut raw = Vec::with_capacity(nl);
            for _ in 0..nl {
                let nkb = rd.count("kink bricks", COUNT_CAP)?;
                let mut lv = Vec::with_capacity(nkb);
                for _ in 0..nkb {
                    let brick = rd.u32()?;
                    let mut m = [0u32; 16];
                    for w in &mut m {
                        *w = rd.u32()?;
                    }
                    let first = rd.u32()?;
                    lv.push((brick, m, first));
                }
                raw.push(lv);
            }
            let ncell = rd.count("cell records", COUNT_CAP)?;
            let mut cells = Vec::with_capacity(ncell);
            for _ in 0..ncell {
                let (off, ntri, nseg) = (rd.u32()?, rd.u16()?, rd.u16()?);
                let flags = if v2 { rd.u32()? } else { CELL_DB };
                if flags == 0 || flags > (CELL_DB | CELL_S | CELL_ATTR) {
                    return Err(bad(format!("cell flags {flags}")));
                }
                cells.push(Cell {
                    off,
                    ntri,
                    nseg,
                    flags,
                });
            }
            let nref = rd.count("refs", COUNT_CAP)?;
            let refs = (0..nref)
                .map(|_| rd.u32())
                .collect::<io::Result<Vec<_>>>()?;
            for (i, c) in cells.iter().enumerate() {
                let end = c.off as usize + c.ntri as usize + c.nseg as usize;
                if end > nref {
                    return Err(bad(format!(
                        "cell record {i} reads refs up to {end} of {nref}"
                    )));
                }
                let (t, s) = refs[c.off as usize..end].split_at(c.ntri as usize);
                if t.iter().any(|&f| f as usize >= nf) || s.iter().any(|&g| g as usize >= nseg) {
                    return Err(bad(format!(
                        "cell record {i}: a triangle or segment ref is out of range"
                    )));
                }
            }
            for lv in raw {
                let mut kl = KinkLevel::default();
                for (brick, m, first) in lv {
                    let ones: u32 = m.iter().map(|x| x.count_ones()).sum();
                    if (first + ones) as usize > ncell {
                        return Err(bad(format!(
                            "brick {brick}: cells {first}+{ones} of {ncell}"
                        )));
                    }
                    let b = brick as usize;
                    if b >= COUNT_CAP {
                        return Err(bad(format!("brick {brick}")));
                    }
                    if kl.slot.len() <= b {
                        kl.slot.resize(b + 1, u32::MAX);
                    }
                    if kl.slot[b] != u32::MAX {
                        return Err(bad(format!("brick {brick} listed twice")));
                    }
                    kl.slot[b] = kl.masks.len() as u32;
                    kl.masks.push(m);
                    kl.first.push(first);
                }
                levels.push(kl);
            }
            let has = !cells.is_empty();
            sheets.push(has.then(|| {
                Arc::new(KinkSheet {
                    slab,
                    origin,
                    verts,
                    faces,
                    rim_vn,
                    vn,
                    face_uv,
                    face_nrm,
                    segs,
                    levels,
                    cells,
                    refs,
                })
            }));
        }
        Ok(Self { sheets })
    }

    /// Number of sheets the file describes.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.sheets.len()
    }

    /// Whether the file describes no sheet.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.sheets.is_empty()
    }

    /// The most triangles and the most segments one kink cell of the file refers to.
    #[must_use]
    pub fn max_cell_refs(&self) -> (u32, u32) {
        self.sheets
            .iter()
            .flatten()
            .map(|k| k.max_cell_refs())
            .fold((0, 0), |(t, s), (a, b)| (t.max(a), s.max(b)))
    }

    /// Attach the kink cells of file sheets `first..first + sheets.len()` to `sheets`. Refused when the file has fewer
    /// sheets, or a sheet's kink levels or bricks do not exist in the sheet.
    pub fn attach(&self, sheets: &mut [super::Sheet], first: usize) -> Result<(), KinkError> {
        if first + sheets.len() > self.sheets.len() {
            return Err(KinkError::Format(format!(
                "sheets {first}..{} of {}",
                first + sheets.len(),
                self.sheets.len()
            )));
        }
        for (i, sh) in sheets.iter_mut().enumerate() {
            let k = first + i;
            let ks = self.sheets[k].clone();
            if let Some(ks) = &ks {
                if ks.levels.len() != sh.levels.len() {
                    return Err(KinkError::Format(format!(
                        "sheet {k}: {} kink levels, the sheet has {}",
                        ks.levels.len(),
                        sh.levels.len()
                    )));
                }
                for (li, (kl, lv)) in ks.levels.iter().zip(&sh.levels).enumerate() {
                    let active = lv.index.iter().filter(|&&x| x >= 0).count();
                    if kl.slot.len() > active {
                        return Err(KinkError::Format(format!(
                            "sheet {k} level {li}: kink brick {} of {active} active",
                            kl.slot.len() - 1
                        )));
                    }
                }
            }
            sh.kink = ks;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn closest_features_of_a_right_triangle() {
        // oracle: the regions of triangle (0,0,0) (1,0,0) (0,1,0) by inspection
        let (a, b, c) = (DVec3::ZERO, DVec3::X, DVec3::Y);
        let cases = [
            (
                DVec3::new(0.2, 0.2, 1.0),
                DVec3::new(0.2, 0.2, 0.0),
                Feature::Face,
            ),
            (DVec3::new(-1.0, -1.0, 0.0), a, Feature::Vertex(0)),
            (DVec3::new(2.0, -0.5, 0.0), b, Feature::Vertex(1)),
            (DVec3::new(-0.5, 2.0, 0.0), c, Feature::Vertex(2)),
            (
                DVec3::new(0.5, -1.0, 0.3),
                DVec3::new(0.5, 0.0, 0.0),
                Feature::Edge(0),
            ),
            (
                DVec3::new(1.0, 1.0, 0.0),
                DVec3::new(0.5, 0.5, 0.0),
                Feature::Edge(1),
            ),
            (
                DVec3::new(-1.0, 0.5, 0.0),
                DVec3::new(0.0, 0.5, 0.0),
                Feature::Edge(2),
            ),
        ];
        for (p, q, f) in cases {
            let (qq, ff) = closest_on_tri(p, a, b, c);
            assert!((qq - q).length() < 1e-12, "{p}: {qq} vs {q}");
            assert_eq!(ff, f, "{p}");
        }
    }

    fn one_triangle(rim: u8, rim_vertices: &[u32], segs: Vec<[u32; 2]>) -> KinkSheet {
        let verts = vec![DVec3::ZERO, DVec3::X, DVec3::Y];
        let mut rim_vn = vec![None; 3];
        for &v in rim_vertices {
            rim_vn[v as usize] = Some(DVec3::Z);
        }
        KinkSheet {
            slab: 0.1,
            origin: DVec3::ZERO,
            verts,
            faces: vec![Face {
                v: [0, 1, 2],
                n: DVec3::Z,
                rim,
                epn: [DVec3::Z; 3],
            }],
            rim_vn,
            vn: vec![DVec3::Z; 3],
            face_uv: Vec::new(),
            face_nrm: Vec::new(),
            segs,
            levels: Vec::new(),
            cells: Vec::new(),
            refs: Vec::new(),
        }
    }

    fn db(ks: &KinkSheet, p: DVec3, tris: &[u32], segs: &[u32]) -> f64 {
        ks.exact_db(p, ks.nearest(p, tris).unwrap(), segs)
    }

    fn sv(ks: &KinkSheet, p: DVec3, tris: &[u32]) -> f64 {
        ks.exact_s(p, ks.nearest(p, tris).unwrap())
    }

    #[test]
    fn exact_db_by_closed_form() {
        // oracle: the definition evaluated by hand on one triangle whose edge ab (y = 0) is the rim
        let ks = one_triangle(0b001, &[0, 1], vec![[0, 1]]);
        // beyond the rim edge ab, 0.3 below the plane line: d_b = in-plane distance to the edge = 0.25
        let v = db(&ks, DVec3::new(0.5, -0.25, 0.3), &[0], &[0]);
        assert!((v - 0.25).abs() < 1e-12, "{v}");
        // inside, 0.2 from the rim, on the sheet: d_b = -min(0.2, H - 0) = -0.1 (the slab H = 0.1 clamps)
        let v = db(&ks, DVec3::new(0.3, 0.2, 0.0), &[0], &[0]);
        assert!((v + 0.1).abs() < 1e-12, "{v}");
        // inside, 0.05 from the rim, 0.02 off the sheet: -min(0.05, 0.1 - 0.02) = -0.05
        let v = db(&ks, DVec3::new(0.3, 0.05, 0.02), &[0], &[0]);
        assert!((v + 0.05).abs() < 1e-12, "{v}");
        // the cell lists no segment: dist_to_rim = 1e3, so the slab answers: -(0.1 - 0.02)
        let v = db(&ks, DVec3::new(0.3, 0.05, 0.02), &[0], &[]);
        assert!((v + 0.08).abs() < 1e-12, "{v}");
        // edge bc is not a rim edge: a point beyond it is inside (closest feature on a non-rim edge)
        let v = db(&ks, DVec3::new(1.0, 1.0, 0.0), &[0], &[0]);
        assert!(v <= 0.0, "{v}");
        // beyond the rim vertex b: pseudonormal z, in-plane distance from b
        let v = db(&ks, DVec3::new(1.3, -0.4, 0.7), &[0], &[0]);
        assert!((v - 0.5).abs() < 1e-12, "{v}");
    }

    #[test]
    fn exact_s_by_closed_form() {
        // oracle: the definition by hand on a fold of two faces sharing edge (0,0,0)-(1,0,0): face A in z = 0 (y > 0, normal
        // +z), face B rising at 90 degrees (z > 0 half-plane x-z at y = 0... normal +y would put it behind A), here B = the
        // triangle (0,0,0) (1,0,0) (0,0,-1)... B lies in y = 0 with normal -y; edge pseudonormal = normalize(+z - y)
        let verts = vec![DVec3::ZERO, DVec3::X, DVec3::Y, DVec3::new(0.0, 0.0, -1.0)];
        let epn = (DVec3::Z - DVec3::Y).normalize();
        let ks = KinkSheet {
            slab: 0.1,
            origin: DVec3::ZERO,
            verts,
            faces: vec![
                Face {
                    v: [0, 1, 2],
                    n: DVec3::Z,
                    rim: 0,
                    epn: [DVec3::Z, DVec3::Z, epn],
                },
                Face {
                    v: [1, 0, 3],
                    n: -DVec3::Y,
                    rim: 0,
                    epn: [-DVec3::Y, -DVec3::Y, epn],
                },
            ],
            rim_vn: vec![None; 4],
            vn: vec![epn; 4],
            face_uv: Vec::new(),
            face_nrm: Vec::new(),
            segs: Vec::new(),
            levels: Vec::new(),
            cells: Vec::new(),
            refs: Vec::new(),
        };
        // above face A's interior: s = +0.3
        assert!((sv(&ks, DVec3::new(0.3, 0.2, 0.3), &[0, 1]) - 0.3).abs() < 1e-12);
        // below face A, but nearer face A than B: s = -0.05
        assert!((sv(&ks, DVec3::new(0.3, 0.2, -0.05), &[0, 1]) + 0.05).abs() < 1e-12);
        // beyond the fold edge (y < 0, z > 0): closest feature the edge, signed by its pseudonormal (+): s = +|r|
        let p = DVec3::new(0.5, -0.3, 0.4);
        assert!(
            (sv(&ks, p, &[0, 1]) - 0.5).abs() < 1e-12,
            "{}",
            sv(&ks, p, &[0, 1])
        );
        // inside the fold (y > 0, z < 0) the nearest feature is a face interior (B at 0.3, A at 0.4): s = r.n_B = -0.3
        let p = DVec3::new(0.5, 0.3, -0.4);
        assert!(
            (sv(&ks, p, &[0, 1]) + 0.3).abs() < 1e-12,
            "{}",
            sv(&ks, p, &[0, 1])
        );
        // a rim edge answers r.n instead (signed, not |r|): mark A's edge ab as rim and probe beyond it in A's plane + 0.1
        let mut kr = ks;
        kr.faces[0].rim = 0b001;
        kr.faces[1].rim = 0b001;
        let p = DVec3::new(0.5, -0.3, 0.1);
        let v = sv(&kr, p, &[0]);
        assert!((v - 0.1).abs() < 1e-12, "{v}");
    }
}

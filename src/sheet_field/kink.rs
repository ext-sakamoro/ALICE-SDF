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
//! On disk (the ASKC sidecar, little-endian; the ASHF file is unchanged): `b"ASKC"`, `u32` version 1, `u32` sheets
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
/// The ASKC version this reader takes.
pub const ASKC_VERSION: u32 = 1;
/// `dist_to_rim` when a cell lists no rim segment (the builder's value).
const NO_RIM: f64 = 1e3;

/// Why an ASKC file was refused.
#[derive(Debug)]
pub enum KinkError {
    /// Reading failed.
    Io(io::Error),
    /// The file is not a valid ASKC sidecar for the field it is attached to.
    Format(String),
}

impl std::fmt::Display for KinkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "ASKC i/o: {e}"),
            Self::Format(s) => write!(f, "not a valid ASKC sidecar: {s}"),
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
}

#[derive(Debug, Clone, Copy)]
struct Cell {
    off: u32,
    ntri: u16,
    nseg: u16,
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

    /// Whether (level, active brick, cell) is a kink cell.
    #[must_use]
    pub fn is_kink(&self, level: usize, brick: usize, cell: usize) -> bool {
        self.cell(level, brick, cell).is_some()
    }

    /// The exact `d_b` at `p` when `p`'s cell (level, active brick, cell index `x*64 + y*8 + z`) is a kink cell.
    #[must_use]
    pub fn db_at(&self, level: usize, brick: usize, cell: usize, p: Vec3) -> Option<f32> {
        let c = self.cell(level, brick, cell)?;
        let off = c.off as usize;
        let tris = &self.refs[off..off + c.ntri as usize];
        let segs = &self.refs[off + c.ntri as usize..off + c.ntri as usize + c.nseg as usize];
        Some(self.exact_db(p.as_dvec3() - self.origin, tris, segs) as f32)
    }

    /// The exact `d_b` at local point `p` from the given triangles and rim segments (module docs).
    fn exact_db(&self, p: DVec3, tris: &[u32], segs: &[u32]) -> f64 {
        let mut best = f64::INFINITY;
        let mut hit: Option<(usize, DVec3, Feature)> = None;
        for &f in tris {
            let fc = &self.faces[f as usize];
            let [a, b, c] = fc.v.map(|i| self.verts[i as usize]);
            let (q, feat) = closest_on_tri(p, a, b, c);
            let d = (p - q).length();
            // the first of equal distances wins (the builder's order: strictly nearer by more than 1e-15)
            if d < best - 1e-15 {
                best = d;
                hit = Some((f as usize, q, feat));
            }
        }
        let Some((f, q, feat)) = hit else {
            // no triangle listed: the cell lies beyond the slab of every triangle (not produced by the builder)
            return NO_RIM;
        };
        let fc = &self.faces[f];
        let r = p - q;
        let boundary_normal = match feat {
            Feature::Edge(e) if fc.rim >> e & 1 == 1 => Some(fc.n),
            Feature::Vertex(i) => self.rim_vn[fc.v[i as usize] as usize],
            _ => None,
        };
        if let Some(n) = boundary_normal {
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
        if version != ASKC_VERSION {
            return Err(KinkError::Format(format!(
                "version {version}, this reader takes {ASKC_VERSION}"
            )));
        }
        let ns = rd.count("sheets", COUNT_CAP)?;
        let mut sheets = Vec::with_capacity(ns);
        for k in 0..ns {
            let bad = |s: String| KinkError::Format(format!("sheet {k}: {s}"));
            let slab = f64::from(rd.f32()?);
            let origin = DVec3::new(rd.f64()?, rd.f64()?, rd.f64()?);
            let nv = rd.count("vertices", COUNT_CAP)?;
            let verts = (0..nv).map(|_| rd.v3()).collect::<io::Result<Vec<_>>>()?;
            let nf = rd.count("faces", COUNT_CAP)?;
            let mut faces = Vec::with_capacity(nf);
            for _ in 0..nf {
                let v = [rd.u32()?, rd.u32()?, rd.u32()?];
                let n = rd.v3()?;
                let rim = rd.u32()?;
                if v.iter().any(|&i| i as usize >= nv) || rim > 7 {
                    return Err(bad(format!("face {v:?} rim {rim} (vertices {nv})")));
                }
                faces.push(Face {
                    v,
                    n,
                    rim: rim as u8,
                });
            }
            let nrv = rd.count("rim vertices", COUNT_CAP)?;
            let mut rim_vn = vec![None; nv];
            for _ in 0..nrv {
                let v = rd.u32()? as usize;
                let n = rd.v3()?;
                if v >= nv {
                    return Err(bad(format!("rim vertex {v} (vertices {nv})")));
                }
                rim_vn[v] = Some(n / n.length().max(1e-12));
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
                cells.push(Cell {
                    off: rd.u32()?,
                    ntri: rd.u16()?,
                    nseg: rd.u16()?,
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
            }],
            rim_vn,
            segs,
            levels: Vec::new(),
            cells: Vec::new(),
            refs: Vec::new(),
        }
    }

    #[test]
    fn exact_db_by_closed_form() {
        // oracle: the definition evaluated by hand on one triangle whose edge ab (y = 0) is the rim
        let ks = one_triangle(0b001, &[0, 1], vec![[0, 1]]);
        // beyond the rim edge ab, 0.3 below the plane line: d_b = in-plane distance to the edge = 0.25
        let v = ks.exact_db(DVec3::new(0.5, -0.25, 0.3), &[0], &[0]);
        assert!((v - 0.25).abs() < 1e-12, "{v}");
        // inside, 0.2 from the rim, on the sheet: d_b = -min(0.2, H - 0) = -0.1 (the slab H = 0.1 clamps)
        let v = ks.exact_db(DVec3::new(0.3, 0.2, 0.0), &[0], &[0]);
        assert!((v + 0.1).abs() < 1e-12, "{v}");
        // inside, 0.05 from the rim, 0.02 off the sheet: -min(0.05, 0.1 - 0.02) = -0.05
        let v = ks.exact_db(DVec3::new(0.3, 0.05, 0.02), &[0], &[0]);
        assert!((v + 0.05).abs() < 1e-12, "{v}");
        // the cell lists no segment: dist_to_rim = 1e3, so the slab answers: -(0.1 - 0.02)
        let v = ks.exact_db(DVec3::new(0.3, 0.05, 0.02), &[0], &[]);
        assert!((v + 0.08).abs() < 1e-12, "{v}");
        // edge bc is not a rim edge: a point beyond it is inside (closest feature on a non-rim edge)
        let v = ks.exact_db(DVec3::new(1.0, 1.0, 0.0), &[0], &[0]);
        assert!(v <= 0.0, "{v}");
        // beyond the rim vertex b: pseudonormal z, in-plane distance from b
        let v = ks.exact_db(DVec3::new(1.3, -0.4, 0.7), &[0], &[0]);
        assert!((v - 0.5).abs() < 1e-12, "{v}");
    }
}

//! Building the kink cells of a sheet and writing the ASKC sidecar ([`super::kink`]), shared by every builder.
//!
//! Input: the sheet's own local mesh (vertices, faces in winding order, the builder's rim edges = its `d_b` boundary,
//! the slab `H`) and the built [`Sheet`] (its levels and stored `s` nodes). The normals are derived here, one definition
//! for every builder (Bærentzen & Aanæs): face normal = normalized `cross(v1 - v0, v2 - v0)`; edge pseudonormal = the sum
//! of the adjacent faces' normals (a boundary edge has one), per face by the opposite corner; vertex pseudonormal = the
//! corner-angle-weighted sum of the incident faces' normals (both unnormalized, as the builder holds them).
//!
//! Two covers, both chosen by shape (no sampling):
//! - `d_b` cells: every cell of the answering level the rim crosses (each rim segment walked at `h_min / 8`, with its
//!   ends), and its 26 neighbours on that level.
//! - s cells: every triangle is rasterised into the cells of every level whose brick is active (triangle-box
//!   separating-axis test). For a surface cell with stored corner values `s_i`, every face `f` crossing it gives
//!   `dev_f = max_i |s_i - n_f.(node_i - a_f)| + q0 / 2` (trilinear reproduces the plane, so `|interp(x)| <= dev_f` for
//!   `x` on `f` in the cell) and `g_f = sum_k min(n_fk lo_k, n_fk hi_k)` (`lo`, `hi`: the min / max of the cell's 4 edge
//!   differences over `h` per axis, the bounds of the interpolant's gradient), and the cell is an s cell when for
//!   some `f` `g_f <= 0` or `dev_f / g_f > eps`: the interpolant's zero may lie farther than `eps` from the surface.
//!
//! References per cell (centre `c`, diagonal `D`, `d_c` = distance from `c` to the sheet, `rho` = distance from `c` to
//! the rim): triangles whose bounding sphere (centroid, farthest corner) reaches within `R1 = d_c + D`; for `d_b`
//! cells also the rim segments whose sphere reaches within `R1 + min(rho + R1, H)`. Every triangle and rim segment the
//! exact value of any point of the cell can use is among them.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::{self, Write};

use glam::DVec3;

use super::kink::{CELL_DB, CELL_S};
use super::{Sheet, BRICK};

/// One sheet's own mesh, as its builder holds it.
#[derive(Debug, Clone, Copy)]
pub struct SheetGeom<'a> {
    /// Vertices (world mm).
    pub verts: &'a [DVec3],
    /// Faces in winding order (the face normal's side).
    pub faces: &'a [[u32; 3]],
    /// The builder's rim edges (its `d_b` boundary), any vertex order.
    pub rim: &'a [[u32; 2]],
    /// The slab `H` of the inside `d_b`.
    pub slab: f64,
}

/// What [`kink_sheet`] found, for the build report.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct KinkStats {
    /// Surface cells (summed over levels).
    pub surface: usize,
    /// s cells.
    pub s_cells: usize,
    /// Of them, cells flagged because the slope bound was not positive.
    pub s_slope_le0: usize,
    /// `d_b` cells.
    pub db_cells: usize,
    /// Cells with both flags.
    pub both: usize,
    /// References (triangles and segments).
    pub refs: usize,
}

/// A sheet's kink cells in file form (the ASKC v2 record of one sheet).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct KinkRecord {
    slab: f64,
    origin: DVec3,
    verts: Vec<[f32; 3]>,
    vn: Vec<[f32; 3]>,
    rim_v: Vec<bool>,
    faces: Vec<[u32; 3]>,
    fnorm: Vec<[f32; 3]>,
    rim_mask: Vec<u32>,
    epn: Vec<[[f32; 3]; 3]>,
    segs: Vec<[u32; 2]>,
    /// Per level: (brick, mask, first) in brick order.
    levels: Vec<Vec<(u32, [u32; 16], u32)>>,
    /// (off, ntri, nseg, flags).
    cells: Vec<(u32, u16, u16, u32)>,
    refs: Vec<u32>,
}

const fn key(a: u32, b: u32) -> (u32, u32) {
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}

fn closest_dist(p: DVec3, a: DVec3, b: DVec3, c: DVec3) -> f64 {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let (d1, d2) = (ab.dot(ap), ac.dot(ap));
    if d1 <= 0.0 && d2 <= 0.0 {
        return ap.length();
    }
    let bp = p - b;
    let (d3, d4) = (ab.dot(bp), ac.dot(bp));
    if d3 >= 0.0 && d4 <= d3 {
        return bp.length();
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        return (p - (a + ab * (d1 / (d1 - d3)))).length();
    }
    let cp = p - c;
    let (d5, d6) = (ab.dot(cp), ac.dot(cp));
    if d6 >= 0.0 && d5 <= d6 {
        return cp.length();
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        return (p - (a + ac * (d2 / (d2 - d6)))).length();
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        return (p - (b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6))))).length();
    }
    let den = 1.0 / (va + vb + vc);
    (p - (a + ab * (vb * den) + ac * (vc * den))).length()
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

/// Triangle vs axis-aligned cube (centre `c`, half size `e`), separating axes (Akenine-Möller).
fn tri_box(v: [DVec3; 3], c: DVec3, e: f64) -> bool {
    let [a, b, d] = v.map(|x| x - c);
    for k in 0..3 {
        let (lo, hi) = (a[k].min(b[k]).min(d[k]), a[k].max(b[k]).max(d[k]));
        if lo > e || hi < -e {
            return false;
        }
    }
    let n = (v[1] - v[0]).cross(v[2] - v[0]);
    if a.dot(n).abs() > e * n.abs().element_sum() + 1e-12 {
        return false;
    }
    for ed in [v[1] - v[0], v[2] - v[1], v[0] - v[2]] {
        for k in 0..3 {
            let mut u = DVec3::ZERO;
            u[k] = 1.0;
            let ax = u.cross(ed);
            if ax == DVec3::ZERO {
                continue;
            }
            let (pa, pb, pd) = (a.dot(ax), b.dot(ax), d.dot(ax));
            let r = e * ax.abs().element_sum();
            if pa.min(pb).min(pd) > r + 1e-12 || pa.max(pb).max(pd) < -r - 1e-12 {
                return false;
            }
        }
    }
    true
}

/// The normals of [module docs](self): face normals, edge pseudonormals by the opposite corner, vertex pseudonormals.
#[must_use]
pub fn derived_normals(geom: &SheetGeom) -> (Vec<DVec3>, Vec<[DVec3; 3]>, Vec<DVec3>) {
    let (vs, fs) = (geom.verts, geom.faces);
    let fnrm: Vec<DVec3> = fs
        .iter()
        .map(|t| {
            let [a, b, c] = t.map(|i| vs[i as usize]);
            (b - a).cross(c - a).normalize()
        })
        .collect();
    let mut en: HashMap<(u32, u32), DVec3> = HashMap::new();
    let mut vn = vec![DVec3::ZERO; vs.len()];
    for (i, t) in fs.iter().enumerate() {
        for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            *en.entry(key(a, b)).or_insert(DVec3::ZERO) += fnrm[i];
        }
        for j in 0..3 {
            let e1 = vs[t[(j + 1) % 3] as usize] - vs[t[j] as usize];
            let e2 = vs[t[(j + 2) % 3] as usize] - vs[t[j] as usize];
            let ang =
                alice_det_math::acos64((e1.dot(e2) / (e1.length() * e2.length())).clamp(-1.0, 1.0));
            vn[t[j] as usize] += ang * fnrm[i];
        }
    }
    let epn: Vec<[DVec3; 3]> = fs
        .iter()
        .map(|t| core::array::from_fn(|j| en[&key(t[(j + 1) % 3], t[(j + 2) % 3])]))
        .collect();
    (fnrm, epn, vn)
}

/// The level's active brick number at brick coordinates `b`, if any.
fn brick_at(lv: &super::Level, b: [i64; 3]) -> Option<usize> {
    if b.iter()
        .zip(lv.nb.iter())
        .any(|(&x, &n)| x < 0 || x >= i64::from(n))
    {
        return None;
    }
    let (x, y, z) = (b[0] as usize, b[1] as usize, b[2] as usize);
    let k = lv.index[(x * lv.nb[1] as usize + y) * lv.nb[2] as usize + z];
    (k >= 0).then_some(k as usize)
}

const fn floor_div(a: i64, b: i64) -> i64 {
    a.div_euclid(b)
}

/// Build one sheet's kink cells ([module docs](self)). `None` when the sheet has no flagged cell.
#[must_use]
pub fn kink_sheet(geom: &SheetGeom, sheet: &Sheet, eps: f64) -> (Option<KinkRecord>, KinkStats) {
    let (vs, fs) = (geom.verts, geom.faces);
    let nf = fs.len();
    let mut st = KinkStats::default();
    if nf == 0 {
        return (None, st);
    }
    let (fnrm, epn, vn) = derived_normals(geom);
    let rimset: HashSet<(u32, u32)> = geom.rim.iter().map(|e| key(e[0], e[1])).collect();
    let mut rim_v = vec![false; vs.len()];
    for e in geom.rim {
        rim_v[e[0] as usize] = true;
        rim_v[e[1] as usize] = true;
    }
    let levels = &sheet.levels;
    let b = BRICK as i64;
    // d_b cells: X+1 around the rim
    let mut cells: BTreeMap<(usize, [i64; 3]), u32> = BTreeMap::new();
    if !geom.rim.is_empty() {
        let hmin = levels
            .iter()
            .map(|l| f64::from(l.h))
            .fold(f64::INFINITY, f64::min);
        for e in geom.rim {
            let (pa, pb) = (vs[e[0] as usize], vs[e[1] as usize]);
            let n = (((pb - pa).length() / (hmin / 8.0)).ceil() as usize + 1).max(2);
            for i in 0..n {
                let p = pa + (pb - pa) * (i as f64 / (n - 1) as f64);
                for (li, lv) in levels.iter().enumerate() {
                    let (h, lo) = (f64::from(lv.h), lv.lo.as_dvec3());
                    let u = ((p - lo) / h).floor();
                    let g = [u.x as i64, u.y as i64, u.z as i64];
                    if brick_at(lv, g.map(|x| floor_div(x, b))).is_none() {
                        continue;
                    }
                    let prior = levels[..li].iter().any(|m| {
                        let w = ((p - m.lo.as_dvec3()) / (f64::from(m.h) * BRICK as f64)).floor();
                        brick_at(m, [w.x as i64, w.y as i64, w.z as i64]).is_some()
                    });
                    if prior {
                        continue;
                    }
                    for dx in -1..=1 {
                        for dy in -1..=1 {
                            for dz in -1..=1 {
                                let gg = [g[0] + dx, g[1] + dy, g[2] + dz];
                                if brick_at(lv, gg.map(|x| floor_div(x, b))).is_some() {
                                    *cells.entry((li, gg)).or_insert(0) |= CELL_DB;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    // s cells: the raster of every triangle, flagged by the position bound
    for (li, lv) in levels.iter().enumerate() {
        let (h, lo) = (f64::from(lv.h), lv.lo.as_dvec3());
        let mut pairs: BTreeMap<[i64; 3], Vec<usize>> = BTreeMap::new();
        for (f, t) in fs.iter().enumerate() {
            let v = t.map(|i| vs[i as usize]);
            let mn = ((v[0].min(v[1]).min(v[2]) - lo) / h).floor();
            let mx = ((v[0].max(v[1]).max(v[2]) - lo) / h).floor();
            for x in mn.x as i64..=mx.x as i64 {
                for y in mn.y as i64..=mx.y as i64 {
                    for z in mn.z as i64..=mx.z as i64 {
                        let g = [x, y, z];
                        if brick_at(lv, g.map(|q| floor_div(q, b))).is_none() {
                            continue;
                        }
                        let c = lo + (DVec3::new(x as f64, y as f64, z as f64) + 0.5) * h;
                        if tri_box(v, c, h / 2.0) {
                            pairs.entry(g).or_default().push(f);
                        }
                    }
                }
            }
        }
        st.surface += pairs.len();
        let ch = &lv.channels[0];
        for (g, flist) in &pairs {
            let k = brick_at(lv, g.map(|q| floor_div(q, b))).expect("active");
            let row = ch.rows[k];
            if row < 0 {
                // a surface cell whose brick stores no s: no certified bound, flag it
                *cells.entry((li, *g)).or_insert(0) |= CELL_S;
                st.s_slope_le0 += 1;
                continue;
            }
            let r = row as usize;
            let (sc, off) = (f64::from(ch.scale[r]), f64::from(ch.offset[r]));
            let cc = g.map(|q| q - floor_div(q, b) * b);
            let mut s8 = [0.0f64; 8];
            let mut nodes = [DVec3::ZERO; 8];
            for (i, (sv, nd)) in s8.iter_mut().zip(nodes.iter_mut()).enumerate() {
                let (dx, dy, dz) = ((i >> 2) & 1, (i >> 1) & 1, i & 1);
                let n = super::NODES;
                let qi = r * n * n * n
                    + ((cc[0] as usize + dx) * n + cc[1] as usize + dy) * n
                    + cc[2] as usize
                    + dz;
                *sv = f64::from(ch.q[qi]) * sc + off;
                *nd = lo
                    + DVec3::new(
                        (g[0] + dx as i64) as f64,
                        (g[1] + dy as i64) as f64,
                        (g[2] + dz as i64) as f64,
                    ) * h;
            }
            let at = |dx: usize, dy: usize, dz: usize| s8[(dx << 2) | (dy << 1) | dz];
            let mut glo = [f64::INFINITY; 3];
            let mut ghi = [f64::NEG_INFINITY; 3];
            for u in 0..2 {
                for w in 0..2 {
                    let d = [
                        (at(1, u, w) - at(0, u, w)) / h,
                        (at(u, 1, w) - at(u, 0, w)) / h,
                        (at(u, w, 1) - at(u, w, 0)) / h,
                    ];
                    for k3 in 0..3 {
                        glo[k3] = glo[k3].min(d[k3]);
                        ghi[k3] = ghi[k3].max(d[k3]);
                    }
                }
            }
            let mut flag = false;
            let mut slope = false;
            for &f in flist {
                let n = fnrm[f];
                let a = vs[fs[f][0] as usize];
                let dev = s8
                    .iter()
                    .zip(&nodes)
                    .map(|(sv, nd)| (sv - n.dot(*nd - a)).abs())
                    .fold(0.0, f64::max)
                    + sc / 2.0;
                let gf: f64 = (0..3)
                    .map(|k3| (n[k3] * glo[k3]).min(n[k3] * ghi[k3]))
                    .sum();
                if gf <= 0.0 {
                    flag = true;
                    slope = true;
                } else if dev / gf > eps {
                    flag = true;
                }
            }
            if flag {
                *cells.entry((li, *g)).or_insert(0) |= CELL_S;
                if slope {
                    st.s_slope_le0 += 1;
                }
            }
        }
    }
    if cells.is_empty() {
        return (None, st);
    }
    // order: level, active brick, cell
    let mut cl: Vec<((usize, [i64; 3]), u32)> = cells.into_iter().collect();
    cl.sort_by_key(|((li, g), _)| {
        let k = brick_at(&levels[*li], g.map(|q| floor_div(q, b))).expect("active");
        let cc = g.map(|q| q - floor_div(q, b) * b);
        (*li, k, cc)
    });
    // references
    let tc: Vec<DVec3> = fs
        .iter()
        .map(|t| t.map(|i| vs[i as usize]).iter().sum::<DVec3>() / 3.0)
        .collect();
    let tr: Vec<f64> = fs
        .iter()
        .zip(&tc)
        .map(|(t, c)| {
            t.iter()
                .map(|&i| (vs[i as usize] - *c).length())
                .fold(0.0, f64::max)
        })
        .collect();
    let segs: Vec<[u32; 2]> = geom.rim.to_vec();
    let mut rec = KinkRecord {
        slab: geom.slab,
        ..KinkRecord::default()
    };
    let mut refs = Vec::new();
    let mut recs = Vec::new();
    let mut lvrec: Vec<BTreeMap<usize, Vec<(usize, usize)>>> = vec![BTreeMap::new(); levels.len()];
    for (i, ((li, g), _)) in cl.iter().enumerate() {
        let k = brick_at(&levels[*li], g.map(|q| floor_div(q, b))).expect("active");
        let cc = g.map(|q| q - floor_div(q, b) * b);
        lvrec[*li]
            .entry(k)
            .or_default()
            .push(((cc[0] * 64 + cc[1] * 8 + cc[2]) as usize, i));
    }
    let mut first_of: HashMap<(usize, usize), u32> = HashMap::new();
    for (li, bricks) in lvrec.iter().enumerate() {
        for (k, list) in bricks {
            first_of.insert((li, *k), recs.len() as u32);
            for &(_, i) in list {
                let ((li2, g), flags) = cl[i];
                let lv = &levels[li2];
                let h = f64::from(lv.h);
                let c = lv.lo.as_dvec3()
                    + (DVec3::new(g[0] as f64, g[1] as f64, g[2] as f64) + 0.5) * h;
                let dd = h * 3f64.sqrt();
                let dc = fs
                    .iter()
                    .map(|t| {
                        closest_dist(c, vs[t[0] as usize], vs[t[1] as usize], vs[t[2] as usize])
                    })
                    .fold(f64::INFINITY, f64::min);
                let r1 = dc + dd;
                let tl: Vec<u32> = (0..nf)
                    .filter(|&j| (tc[j] - c).length() <= r1 + tr[j])
                    .map(|j| j as u32)
                    .collect();
                let sl: Vec<u32> = if flags & CELL_DB != 0 {
                    let rho = segs
                        .iter()
                        .map(|e| seg_dist(c, vs[e[0] as usize], vs[e[1] as usize]))
                        .fold(f64::INFINITY, f64::min);
                    let r2 = r1 + (rho + r1).min(geom.slab);
                    segs.iter()
                        .enumerate()
                        .filter(|(_, e)| {
                            let (pa, pb) = (vs[e[0] as usize], vs[e[1] as usize]);
                            ((pa + pb) / 2.0 - c).length() <= r2 + (pa - pb).length() / 2.0
                        })
                        .map(|(j, _)| j as u32)
                        .collect()
                } else {
                    Vec::new()
                };
                recs.push((refs.len() as u32, tl.len() as u16, sl.len() as u16, flags));
                refs.extend(tl);
                refs.extend(sl);
            }
        }
    }
    for (li, bricks) in lvrec.iter().enumerate() {
        let mut lvl = Vec::new();
        for (k, list) in bricks {
            let mut m = [0u32; 16];
            for &(cidx, _) in list {
                m[cidx / 32] |= 1 << (cidx % 32);
            }
            lvl.push((*k as u32, m, first_of[&(li, *k)]));
        }
        rec.levels.push(lvl);
    }
    let org = vs.iter().fold(DVec3::INFINITY, |a, v| a.min(*v)).floor();
    rec.origin = org;
    rec.verts = vs.iter().map(|v| (*v - org).as_vec3().to_array()).collect();
    rec.vn = vn.iter().map(|v| v.as_vec3().to_array()).collect();
    rec.rim_v = rim_v;
    rec.faces = fs.to_vec();
    rec.fnorm = fnrm.iter().map(|v| v.as_vec3().to_array()).collect();
    rec.rim_mask = fs
        .iter()
        .map(|t| {
            u32::from(rimset.contains(&key(t[0], t[1])))
                | (u32::from(rimset.contains(&key(t[1], t[2]))) << 1)
                | (u32::from(rimset.contains(&key(t[2], t[0]))) << 2)
        })
        .collect();
    rec.epn = epn
        .iter()
        .map(|e| e.map(|v| v.as_vec3().to_array()))
        .collect();
    rec.segs = segs;
    st.s_cells = recs.iter().filter(|r| r.3 & CELL_S != 0).count();
    st.db_cells = recs.iter().filter(|r| r.3 & CELL_DB != 0).count();
    st.both = recs.iter().filter(|r| r.3 == CELL_DB | CELL_S).count();
    st.refs = refs.len();
    rec.cells = recs;
    rec.refs = refs;
    (Some(rec), st)
}

/// Write the ASKC v2 sidecar.
///
/// One record per sheet of the ASHF file, in file order, each with its slab and level count (a `None` record: the
/// sheet has no flagged cell, written as empty levels).
pub fn write_askc<W: Write>(
    w: &mut W,
    sheets: &[(Option<KinkRecord>, f64, usize)],
) -> io::Result<u64> {
    let mut n = 0u64;
    let mut put = |w: &mut W, b: &[u8]| -> io::Result<()> {
        n += b.len() as u64;
        w.write_all(b)
    };
    put(w, b"ASKC")?;
    put(w, &2u32.to_le_bytes())?;
    put(w, &(sheets.len() as u32).to_le_bytes())?;
    for (rec, slab, nlev) in sheets {
        let Some(r) = rec else {
            put(w, &(*slab as f32).to_le_bytes())?;
            for _ in 0..3 {
                put(w, &0f64.to_le_bytes())?;
            }
            for _ in 0..3 {
                put(w, &0u32.to_le_bytes())?;
            }
            put(w, &(*nlev as u32).to_le_bytes())?;
            for _ in 0..*nlev {
                put(w, &0u32.to_le_bytes())?;
            }
            put(w, &0u32.to_le_bytes())?;
            put(w, &0u32.to_le_bytes())?;
            continue;
        };
        put(w, &(r.slab as f32).to_le_bytes())?;
        for x in [r.origin.x, r.origin.y, r.origin.z] {
            put(w, &x.to_le_bytes())?;
        }
        put(w, &(r.verts.len() as u32).to_le_bytes())?;
        for ((v, vn), rim) in r.verts.iter().zip(&r.vn).zip(&r.rim_v) {
            for x in v.iter().chain(vn) {
                put(w, &x.to_le_bytes())?;
            }
            put(w, &u32::from(*rim).to_le_bytes())?;
        }
        put(w, &(r.faces.len() as u32).to_le_bytes())?;
        for (((f, nrm), m), e) in r.faces.iter().zip(&r.fnorm).zip(&r.rim_mask).zip(&r.epn) {
            for x in f {
                put(w, &x.to_le_bytes())?;
            }
            for x in nrm {
                put(w, &x.to_le_bytes())?;
            }
            put(w, &m.to_le_bytes())?;
            for x in e.iter().flatten() {
                put(w, &x.to_le_bytes())?;
            }
        }
        put(w, &(r.segs.len() as u32).to_le_bytes())?;
        for s in &r.segs {
            put(w, &s[0].to_le_bytes())?;
            put(w, &s[1].to_le_bytes())?;
        }
        put(w, &(r.levels.len() as u32).to_le_bytes())?;
        for lv in &r.levels {
            put(w, &(lv.len() as u32).to_le_bytes())?;
            for (bk, m, first) in lv {
                put(w, &bk.to_le_bytes())?;
                for x in m {
                    put(w, &x.to_le_bytes())?;
                }
                put(w, &first.to_le_bytes())?;
            }
        }
        put(w, &(r.cells.len() as u32).to_le_bytes())?;
        for (off, nt, ns, fl) in &r.cells {
            put(w, &off.to_le_bytes())?;
            put(w, &nt.to_le_bytes())?;
            put(w, &ns.to_le_bytes())?;
            put(w, &fl.to_le_bytes())?;
        }
        put(w, &(r.refs.len() as u32).to_le_bytes())?;
        for x in &r.refs {
            put(w, &x.to_le_bytes())?;
        }
    }
    Ok(n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sheet_field::kink::KinkSheet;

    #[test]
    fn derived_normals_by_hand_and_the_point_evaluator() {
        // oracle: a unit right triangle in z = 0 (normal +z), one rim edge ab (y = 0); by hand: face n = +z, every edge
        // pseudonormal = +z (one face each), vertex pseudonormals = corner angle x +z (pi/2 at a, pi/4 at b and c)
        let verts = [DVec3::ZERO, DVec3::X, DVec3::Y];
        let faces = [[0u32, 1, 2]];
        let rim = [[0u32, 1]];
        let g = SheetGeom {
            verts: &verts,
            faces: &faces,
            rim: &rim,
            slab: 0.1,
        };
        let (fnrm, epn, vn) = derived_normals(&g);
        assert!((fnrm[0] - DVec3::Z).length() < 1e-12);
        assert!(epn[0].iter().all(|e| (*e - DVec3::Z).length() < 1e-12));
        let q = std::f64::consts::FRAC_PI_4;
        for (v, w) in vn.iter().zip([2.0 * q, q, q]) {
            assert!((*v - w * DVec3::Z).length() < 1e-12, "{v}");
        }
        let ks = KinkSheet::from_geom(&g);
        // beyond the rim edge, 0.3 above the plane: s = r.n = 0.3, d_b = in-plane 0.25, distance 0.39051
        let (s, db, ud) = ks
            .eval_all(DVec3::new(0.5, -0.25, 0.3), &[0], &[0])
            .unwrap();
        assert!(
            (s - 0.3).abs() < 1e-12
                && (db - 0.25).abs() < 1e-12
                && (ud - (0.0625f64 + 0.09).sqrt()).abs() < 1e-12,
            "{s} {db} {ud}"
        );
        // inside, 0.05 from the rim, 0.02 below: s = -0.02, d_b = -min(0.05, 0.1 - 0.02) = -0.05
        let (s, db, _) = ks
            .eval_all(DVec3::new(0.3, 0.05, -0.02), &[0], &[0])
            .unwrap();
        assert!(
            (s + 0.02).abs() < 1e-12 && (db + 0.05).abs() < 1e-12,
            "{s} {db}"
        );
        assert!(ks.eval_all(DVec3::ZERO, &[], &[]).is_none());
    }
}

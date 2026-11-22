//! askc_build <field.ashf> <geom.asgm> <out.askc> [eps]: the kink cells of every sheet of a field (shared cover,
//! `sheet_field::kink_build`) from the builder's per-sheet meshes, written as an ASKC v2 sidecar. The geometry file:
//! b"ASGM", u32 version (1 or 2), u32 sheets; per sheet f64 slab, u32 nv, f64 V[nv*3], u32 nf, u32 F[nf*3], u32 nrim,
//! u32 rim[nrim*2]; v2 then u32 attribute flags (bit 0 corner UVs, bit 1 corner author normals) and per face f64 UV[6] /
//! f64 normal[9] for each flag set.

use std::io::{BufReader, Read};

use alice_sdf::sheet_field::kink::{KinkFile, CELL_DB, CELL_S};
use alice_sdf::sheet_field::kink_build::{kink_sheet, write_askc, KinkStats, SheetGeom};
use alice_sdf::sheet_field::stream::AshfIndex;
use glam::DVec3;

fn u32_at(b: &[u8], o: &mut usize) -> u32 {
    let v = u32::from_le_bytes(b[*o..*o + 4].try_into().expect("4 bytes"));
    *o += 4;
    v
}

fn f64_at(b: &[u8], o: &mut usize) -> f64 {
    let v = f64::from_le_bytes(b[*o..*o + 8].try_into().expect("8 bytes"));
    *o += 8;
    v
}

#[allow(
    clippy::disallowed_methods,
    reason = "the build report prints its wall time; nothing written depends on it"
)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 4 {
        return Err("usage: askc_build <field.ashf> <geom.asgm> <out.askc> [eps]".into());
    }
    let eps: f64 = a.get(4).map_or(Ok(0.01), |s| s.parse())?;
    if std::path::Path::new(&a[3]).exists() {
        return Err(format!("{} exists (never overwritten)", a[3]).into());
    }
    let mut fh = BufReader::new(std::fs::File::open(&a[1])?);
    let ix = AshfIndex::scan(&mut fh)?;
    let mut g = Vec::new();
    std::fs::File::open(&a[2])?.read_to_end(&mut g)?;
    if &g[..4] != b"ASGM" {
        return Err("not an ASGM file".into());
    }
    let mut o = 4;
    let gver = u32_at(&g, &mut o);
    if !(1..=2).contains(&gver) {
        return Err(format!("ASGM version {gver}").into());
    }
    let ns = u32_at(&g, &mut o) as usize;
    let mut sheets = ix.read_sheets(&mut fh, 0..ns)?;
    if sheets.len() != ns {
        return Err(format!("{} sheets in the field, {ns} in the geometry", sheets.len()).into());
    }
    let t0 = std::time::Instant::now();
    let mut out = Vec::with_capacity(ns);
    let mut tot = KinkStats::default();
    for sh in &sheets {
        let slab = f64_at(&g, &mut o);
        let nv = u32_at(&g, &mut o) as usize;
        let verts: Vec<DVec3> = (0..nv)
            .map(|_| DVec3::new(f64_at(&g, &mut o), f64_at(&g, &mut o), f64_at(&g, &mut o)))
            .collect();
        let nf = u32_at(&g, &mut o) as usize;
        let faces: Vec<[u32; 3]> = (0..nf)
            .map(|_| [u32_at(&g, &mut o), u32_at(&g, &mut o), u32_at(&g, &mut o)])
            .collect();
        let nr = u32_at(&g, &mut o) as usize;
        let rim: Vec<[u32; 2]> = (0..nr)
            .map(|_| [u32_at(&g, &mut o), u32_at(&g, &mut o)])
            .collect();
        let attr = if gver >= 2 { u32_at(&g, &mut o) } else { 0 };
        let uvs: Vec<[[f64; 2]; 3]> = if attr & 1 == 1 {
            (0..nf)
                .map(|_| core::array::from_fn(|_| [f64_at(&g, &mut o), f64_at(&g, &mut o)]))
                .collect()
        } else {
            Vec::new()
        };
        let nrms: Vec<[[f64; 3]; 3]> = if attr & 2 == 2 {
            (0..nf)
                .map(|_| {
                    core::array::from_fn(|_| {
                        [f64_at(&g, &mut o), f64_at(&g, &mut o), f64_at(&g, &mut o)]
                    })
                })
                .collect()
        } else {
            Vec::new()
        };
        let geom = SheetGeom {
            verts: &verts,
            faces: &faces,
            rim: &rim,
            slab,
            uv: (attr & 1 == 1).then_some(uvs.as_slice()),
            nrm: (attr & 2 == 2).then_some(nrms.as_slice()),
        };
        let (rec, st) = kink_sheet(&geom, sh, eps);
        tot.surface += st.surface;
        tot.s_cells += st.s_cells;
        tot.s_slope_le0 += st.s_slope_le0;
        tot.db_cells += st.db_cells;
        tot.both += st.both;
        tot.refs += st.refs;
        out.push((rec, slab, sh.levels.len()));
    }
    if o != g.len() {
        return Err(format!("geometry: {} trailing bytes", g.len() - o).into());
    }
    let tmp = format!("{}.tmp", a[3]);
    let n = write_askc(
        &mut std::io::BufWriter::new(std::fs::File::create(&tmp)?),
        &out,
    )?;
    std::fs::rename(&tmp, &a[3])?;
    // read the sidecar back onto the sheets and count its flagged cells and mesh sizes against what was built
    let kink = KinkFile::read(&mut BufReader::new(std::fs::File::open(&a[3])?))?;
    kink.attach(&mut sheets, 0)?;
    let (mut s_cells, mut db_cells, mut faces, mut segments) = (0u64, 0u64, 0usize, 0usize);
    for sh in &sheets {
        let Some(k) = &sh.kink else { continue };
        faces += k.face_count();
        segments += k.segment_count();
        for (l, lv) in sh.levels.iter().enumerate() {
            let active = lv.index.iter().filter(|&&x| x >= 0).count();
            for b in 0..active {
                for c in 0..512 {
                    let f = k.flags(l, b, c).unwrap_or(0);
                    s_cells += u64::from(f & CELL_S != 0);
                    db_cells += u64::from(f & CELL_DB != 0);
                }
            }
        }
    }
    if s_cells != tot.s_cells as u64 || db_cells != tot.db_cells as u64 {
        return Err(format!(
            "read back: s cells {s_cells}, d_b cells {db_cells}; built {} and {}",
            tot.s_cells, tot.db_cells
        )
        .into());
    }
    println!("[askc_build] read back: s cells {s_cells}, d_b cells {db_cells}, {faces} faces, {segments} rim segments");
    println!(
        "[askc_build] {} sheets: surface cells {}, s cells {} ({:.2} %, slope <= 0: {}), d_b cells {}, both {}, refs {}, {:.1} MB, eps {eps}, {:.1} s",
        ns,
        tot.surface,
        tot.s_cells,
        100.0 * tot.s_cells as f64 / tot.surface.max(1) as f64,
        tot.s_slope_le0,
        tot.db_cells,
        tot.both,
        tot.refs,
        n as f64 / 1e6,
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}

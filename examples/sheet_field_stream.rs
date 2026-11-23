//! Sheet field streamed from an ASHF file: a stack of flat squares written to a file, read back by its index,
//! split into chunks by channel bytes and into batches by a host memory budget, and ray-marched one batch at a
//! time. The streamed hits are compared with the same field held in memory. A second field stores its UV channels
//! on a coarse grid (ASHF version 5) and as a band around the surface (ASHF version 6), and the bytes of the three files are
//! printed.
//!
//! Run: `cargo run --example sheet_field_stream --no-default-features`

use alice_sdf::sheet_field::build::{band, coarsen, level_from_fn, sheet};
use alice_sdf::sheet_field::stream::{
    available_memory, chunk_split, host_batches, march_streamed, sheet_payload_bytes, write_ashf,
    AshfIndex,
};
use alice_sdf::sheet_field::{MarchResult, Sheet, SheetField, ValidRule};
use glam::Vec3;
use std::io::{Cursor, Seek, SeekFrom};

const Z0: f32 = 15.0;

/// A flat square `z = z` of half size `half` around (cx, cy), with a UV map when `uv`
fn square(
    z: f32,
    cx: f32,
    cy: f32,
    half: f32,
    uv: bool,
) -> Result<Sheet, Box<dyn std::error::Error>> {
    let lo = Vec3::new(cx - 8.0 + 0.013, cy - 8.0 + 0.021, z - 1.0 + 0.007);
    let level = level_from_fn(
        lo,
        0.25,
        [8, 8, 1],
        |_c, _e| true,
        |p| {
            let (dx, dy) = ((p.x - cx).abs() - half, (p.y - cy).abs() - half);
            let db = Vec3::new(dx.max(0.0), dy.max(0.0), 0.0).length() + dx.max(dy).min(0.0);
            let mut v = vec![p.z - z, db];
            if uv {
                v.extend([(p.x - cx) / 16.0 + 0.5, (p.y - cy) / 16.0 + 0.5]);
            }
            v
        },
    )?;
    Ok(sheet(vec![level])?)
}

fn ashf(field: &SheetField) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut bytes = Vec::new();
    write_ashf(&mut bytes, field)?;
    Ok(bytes)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // eight squares stacked downward, shifted so the lower ones show around the upper ones
    let sheets = (0..8)
        .map(|k| {
            let z = 10.0 - 0.7 * k as f32;
            let (cx, cy) = (((k * 5) % 7) as f32 - 3.0, ((k * 3) % 5) as f32 - 2.0);
            square(z, cx, cy, 2.0 + (k % 4) as f32 * 0.9, k % 3 == 0)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let field = SheetField::new(sheets, ValidRule::CellV1);
    let payload: u64 = field.sheets.iter().map(sheet_payload_bytes).sum();
    let mut file = Cursor::new(ashf(&field)?);
    let index = AshfIndex::scan(&mut file)?;
    println!(
        "ASHF v{}: {} sheets, {} bytes in the file, {payload} bytes of arrays in memory",
        index.version,
        index.spans.len(),
        file.get_ref().len()
    );

    // chunks of about two sheets of channel bytes; batches within the host memory the system reports (or 1 MiB)
    let cap = index
        .spans
        .iter()
        .map(|s| s.channel_bytes)
        .max()
        .unwrap_or(0)
        * 2;
    let chunks = chunk_split(&index, cap, cap)?;
    let one_chunk = chunks
        .iter()
        .map(|c| {
            index.spans[c.clone()]
                .iter()
                .map(|s| s.host_bytes)
                .sum::<u64>()
        })
        .max()
        .unwrap_or(0);
    let budget = available_memory()
        .unwrap_or(1 << 20)
        .min(2 * one_chunk)
        .max(one_chunk);
    let batches = host_batches(&index, &chunks, budget)?;
    let origins: Vec<Vec3> = (0..24 * 24)
        .map(|i| {
            Vec3::new(
                -9.0 + 18.0 * ((i % 24) as f32 + 0.37) / 24.0,
                -9.0 + 18.0 * ((i / 24) as f32 + 0.61) / 24.0,
                Z0,
            )
        })
        .collect();
    file.seek(SeekFrom::Start(0))?;
    let (streamed, stats) = march_streamed(
        &index,
        &mut file,
        &chunks,
        &batches,
        origins.len(),
        |chunk| {
            Ok(origins
                .iter()
                .map(|o| chunk.march_decided(*o, Vec3::NEG_Z, 2.0 * Z0))
                .collect())
        },
    )?;
    let same = origins
        .iter()
        .zip(&streamed)
        .filter(|(o, r)| field.march_decided(**o, Vec3::NEG_Z, 2.0 * Z0) == **r)
        .count();
    let hits = streamed
        .iter()
        .filter(|r| matches!(r, MarchResult::Hit(_)))
        .count();
    println!(
        "streamed: {} chunks in {} batches (budget {budget} bytes, peak {} bytes); {hits} hits; the field in memory \
         gives the same result on {same} of {} rays",
        stats.chunks,
        stats.batches,
        stats.peak_host_bytes,
        origins.len()
    );

    // the UV channels of one square: whole (v1), on a coarse grid (v2), as a band around the surface (v3)
    let whole = square(0.0, 0.0, 0.0, 3.0, true)?;
    let coarse = {
        let mut level = whole.levels[0].clone();
        for channel in level.channels.iter_mut().skip(2) {
            *channel = coarsen(channel, 5)?;
        }
        sheet(vec![level])?
    };
    let banded = sheet(vec![band(&whole.levels[0], 2.0)?])?;
    for (name, s) in [("whole", whole), ("coarse", coarse), ("band", banded)] {
        let bytes = ashf(&SheetField::new(vec![s], ValidRule::CellV1))?;
        let index = AshfIndex::scan(&mut Cursor::new(&bytes))?;
        let back = index.read_sheets(&mut Cursor::new(&bytes), 0..index.spans.len())?;
        println!(
            "UV {name}: ASHF v{}, {} bytes, read back {} sheet(s)",
            index.version,
            bytes.len(),
            back.len()
        );
    }
    Ok(())
}
